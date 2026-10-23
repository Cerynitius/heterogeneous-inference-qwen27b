#!/usr/bin/env python3
"""Derived checkpoint with an NVFP4 W4A16 MTP draft layer.

nvidia/Qwen3.8-27B-NVFP4 keeps the MTP layer in bf16 (~850 MB read per draft pass, excluded via ignore "mtp*").
The draft only changes the acceptance rate (verification is exact), so the draft may use a quantized copy.
W4A16_NVFP4 (ModelOpt MIXED_PRECISION algo, Marlin kernel in vLLM 0.30): fp4 weights, bf16 activations, no
activation calibration. One draft pass: 705 us bf16 -> 220 us Marlin (PRO 5000).

Output dir = source dir with shards 1-2 and small files symlinked; shard 3 rewritten (MTP linears -> uint8 packed e2m1
[N, K/2] + float8_e4m3fn block scales [N, K/16] + float32 weight_scale_2 = amax/2688, ModelOpt layout); config.json and
hf_quant_config.json: MTP removed from ignore/exclude_modules, its 8 linears listed as W4A16_NVFP4 group 16.
Fused shards (q/k/v, gate/up) share one global scale (joint amax), as ModelOpt does for the main layers.
  python tools/build_mtp_w4a16.py <nvidia/Qwen3.8-27B-NVFP4 dir>   # writes <dir>-mtpw4a16
"""
import json, os, sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
from flashinfer import nvfp4_quantize, SfLayout

SRC = os.path.abspath(sys.argv[1]).rstrip("/")
DST = SRC + "-mtpw4a16"
SHARD = "model-00003-of-00003.safetensors"
P = "mtp.layers.0."
GROUPS = [[P + "self_attn.q_proj", P + "self_attn.k_proj", P + "self_attn.v_proj"], [P + "self_attn.o_proj"],
          [P + "mlp.gate_proj", P + "mlp.up_proj"], [P + "mlp.down_proj"], ["mtp.fc"]]
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")

os.makedirs(DST, exist_ok=False)
for f in os.listdir(SRC):
    if f in (SHARD, "config.json", "hf_quant_config.json", "model.safetensors.index.json") or f.startswith("."):
        continue
    os.symlink(os.path.join(SRC, f), os.path.join(DST, f))

with safe_open(os.path.join(SRC, SHARD), "pt") as f:
    meta = f.metadata()
    tensors = {k: f.get_tensor(k) for k in f.keys()}
idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
ref = "model.language_model.layers.0.mlp.gate_proj."
with safe_open(os.path.join(SRC, idx["weight_map"][ref + "weight"]), "pt") as f:
    rdt = {s: f.get_tensor(ref + s) for s in ("weight", "weight_scale", "weight_scale_2")}

report = []
for grp in GROUPS:
    ws = {n: tensors.pop(n + ".weight").cuda() for n in grp}
    amax = max(w.float().abs().max() for w in ws.values())
    gsf = (448 * 6) / amax
    for n, w in ws.items():
        q, sf = nvfp4_quantize(w, gsf, sfLayout=SfLayout.layout_linear, do_shuffle=False)
        sf = sf.view(torch.float8_e4m3fn).reshape(w.shape[0], w.shape[1] // 16)
        lo, hi = E2M1[(q & 0xF).long()], E2M1[(q >> 4).long()]
        dq = torch.stack([lo, hi], -1).reshape(w.shape) * sf.float().repeat_interleave(16, 1) / gsf
        err = ((dq - w.float()).norm() / w.float().norm()).item()
        cos = torch.nn.functional.cosine_similarity(dq.flatten(), w.float().flatten(), dim=0).item()
        report.append(f"{n:34s} {tuple(w.shape)} rel_fro {err:.4f} cos {cos:.5f}")
        tensors[n + ".weight"] = q.cpu().to(rdt["weight"].dtype)
        tensors[n + ".weight_scale"] = sf.cpu().to(rdt["weight_scale"].dtype)
        tensors[n + ".weight_scale_2"] = (1.0 / gsf).float().reshape(rdt["weight_scale_2"].shape).cpu()
        idx["weight_map"][n + ".weight_scale"] = SHARD
        idx["weight_map"][n + ".weight_scale_2"] = SHARD
print("\n".join(report))
save_file(tensors, os.path.join(DST, SHARD), metadata=meta)
json.dump(idx, open(os.path.join(DST, "model.safetensors.index.json"), "w"), indent=2)

layers = {n: {"quant_algo": "W4A16_NVFP4", "group_size": 16} for g in GROUPS for n in g}
cfg = json.load(open(os.path.join(SRC, "config.json")))
qc = cfg["quantization_config"]
qc["ignore"] = [x for x in qc["ignore"] if not x.startswith("mtp")]
qc["quantized_layers"].update(layers)
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=2)
hq = json.load(open(os.path.join(SRC, "hf_quant_config.json")))
hq["quantization"]["exclude_modules"] = [x for x in hq["quantization"]["exclude_modules"] if not x.startswith("mtp")]
hq["quantization"]["quantized_layers"].update(layers)
json.dump(hq, open(os.path.join(DST, "hf_quant_config.json"), "w"), indent=4)
print("written", DST, "ignore:", qc["ignore"], "exclude:", hq["quantization"]["exclude_modules"])
