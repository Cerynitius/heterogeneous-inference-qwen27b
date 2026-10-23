#!/usr/bin/env python3
"""Derived DFlash2 drafter with NVFP4 W4A16 linears, same recipe as build_mtp_w4a16.py.

The drafter only changes the acceptance rate (the target verifies every draft exactly). bf16 DFlash2 reads 3.85 GB per
draft pass (+3-4 ms/step on the PRO 5000 vs MTP); quantizing its 5 layers' o, gate/up, down and fc to
W4A16_NVFP4 (Marlin) cuts that to ~1.1 GB. Conv kernels, the candidate selector (codebooks, hidden_projection) and
norms stay bf16. Fused shards (q/k/v, gate/up) share one global scale (joint amax), as ModelOpt does.
quantized_layers lists every linear under the local name (layers.i), the global vLLM name (model.layers.<64+i>) and
layers.<64+i>, so the MIXED_PRECISION lookup matches whatever prefix the draft model is built with (vLLM #61003 is the
same aliasing problem for FP8 exclusions).
  python tools/build_dflash2_w4a16.py <DFlash2 drafter dir>   # writes <dir>-w4a16
"""
import json, os, sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
from flashinfer import nvfp4_quantize, SfLayout

SRC = os.path.abspath(sys.argv[1]).rstrip("/")
DST = SRC + "-w4a16"
NT = 64                                                     # target layers: draft layer i is built as layers.<64+i>
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
cfg = json.load(open(os.path.join(SRC, "config.json")))
L = cfg["num_hidden_layers"]
GROUPS = [["fc"]]
for i in range(L):
    p = f"layers.{i}."
    # q/k/v stay bf16: DFlash stacks every layer's K/V rows of qkv_proj into one fused bf16 GEMM for the context-KV
    # precompute (qwen3_dflash.py _build_context_kv_buffers), which slices a packed fp4 weight wrongly (shape error).
    GROUPS += [[p + "self_attn.o_proj"], [p + "mlp.gate_proj", p + "mlp.up_proj"], [p + "mlp.down_proj"]]

os.makedirs(DST, exist_ok=False)
for f in os.listdir(SRC):
    if f in ("model.safetensors", "config.json") or f.startswith("."):
        continue
    os.symlink(os.path.join(SRC, f), os.path.join(DST, f))
with safe_open(os.path.join(SRC, "model.safetensors"), "pt") as f:
    meta = f.metadata()
    tensors = {k: f.get_tensor(k) for k in f.keys()}

report, qlayers = [], {}
for grp in GROUPS:
    ws = {n: tensors.pop(n + ".weight").cuda() for n in grp}             # largest group (gate+up) 356 MB
    amax = max(w.float().abs().max() for w in ws.values())
    gsf = (448 * 6) / amax
    for n, w in ws.items():
        q, sf = nvfp4_quantize(w, gsf, sfLayout=SfLayout.layout_linear, do_shuffle=False)
        sf = sf.view(torch.float8_e4m3fn).reshape(w.shape[0], w.shape[1] // 16)
        q, sf, wc, g = q.cpu(), sf.cpu(), w.float().cpu(), gsf.cpu()      # error stats on the CPU (GPU is shared)
        lo, hi = E2M1.cpu()[(q & 0xF).long()], E2M1.cpu()[(q >> 4).long()]
        dq = torch.stack([lo, hi], -1).reshape(w.shape) * sf.float().repeat_interleave(16, 1) / g
        err = ((dq - wc).norm() / wc.norm()).item()
        cos = torch.nn.functional.cosine_similarity(dq.flatten(), wc.flatten(), dim=0).item()
        del dq, lo, hi, wc
        report.append(f"{n:28s} {tuple(w.shape)} rel_fro {err:.4f} cos {cos:.5f}")
        tensors[n + ".weight"] = q
        tensors[n + ".weight_scale"] = sf
        tensors[n + ".weight_scale_2"] = (1.0 / gsf).float().reshape(()).cpu()
        info = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
        names = {n}
        if n.startswith("layers."):
            i, rest = n.split(".", 2)[1:]
            names |= {f"model.layers.{int(i) + NT}.{rest}", f"layers.{int(i) + NT}.{rest}", "model." + n}
        else:
            names |= {"model." + n}
        for nm in names:
            qlayers[nm] = info
    del ws
    torch.cuda.empty_cache()
print("\n".join(report))
save_file(tensors, os.path.join(DST, "model.safetensors"), metadata=meta)
cfg["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                              "quantization": {"quant_algo": "MIXED_PRECISION", "quantized_layers": qlayers}}
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=2)
# vLLM's modelopt_mixed loader reads hf_quant_config.json (config.json alone -> "Cannot find the config file")
json.dump({"producer": {"name": "modelopt", "version": "heterogeneous-inference-qwen27b build_dflash2_w4a16"},
           "quantization": {"quant_algo": "MIXED_PRECISION", "quantized_layers": qlayers}},
          open(os.path.join(DST, "hf_quant_config.json"), "w"), indent=2)
print("wrote", DST, sum(v.numel() * v.element_size() for v in tensors.values()) / 1e9, "GB")
