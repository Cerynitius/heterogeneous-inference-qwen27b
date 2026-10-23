"""Large-image shrinking for the router.

Every OpenAI chat image part (image_url: data URL or http(s) URL) above --img-max-pixels is resized in a worker
process to the size the Qwen3.8 processor would keep under that budget (sides multiples of 32 = patch 16 x merge 2,
so the processor does not resize again) and inlined as a lossless PNG data URL. Orientation and transparency are
normalised exactly as vLLM does it (EXIF transpose, alpha composited on white) before the resize. Images within the
budget are passed through untouched (same bytes -> same vLLM mm hash -> prefix cache hits). Results are cached by
input hash, so a session re-sending the same screenshot every turn pays once.
"""
import asyncio, base64, binascii, hashlib, io, math
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

FACTOR = 32          # patch_size 16 x spatial_merge_size 2
MIN_PIXELS = 65536   # preprocessor_config.json size.shortest_edge (= min pixels)


def smart_resize(h, w, max_pixels):
    """Qwen2-VL / Qwen3-VL processor rule: round to FACTOR, keep the aspect, total within [MIN_PIXELS, max_pixels]."""
    hb = max(FACTOR, round(h / FACTOR) * FACTOR)
    wb = max(FACTOR, round(w / FACTOR) * FACTOR)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb = max(FACTOR, math.floor(h / beta / FACTOR) * FACTOR)
        wb = max(FACTOR, math.floor(w / beta / FACTOR) * FACTOR)
    elif hb * wb < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (h * w))
        hb = math.ceil(h * beta / FACTOR) * FACTOR
        wb = math.ceil(w * beta / FACTOR) * FACTOR
    return hb, wb


def _shrink(data, max_pixels):
    """Worker: returns (png_bytes or None if untouched, width, height of what the model will see)."""
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = 400_000_000
    im = Image.open(io.BytesIO(data))
    w, h = im.size
    if w * h <= max_pixels:
        hb, wb = smart_resize(h, w, max_pixels)
        return None, wb, hb
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    im.load()
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    elif im.mode != "RGB":
        im = im.convert("RGB")
    w, h = im.size
    hb, wb = smart_resize(h, w, max_pixels)
    im = im.resize((wb, hb), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    im.save(out, format="PNG", compress_level=1)
    return out.getvalue(), wb, hb


class Shrinker:
    def __init__(self, max_pixels, workers, cache_bytes, fetch_limit, log):
        self.max_pixels, self.log = max_pixels, log
        self.pool = ProcessPoolExecutor(max_workers=workers) if max_pixels > 0 else None
        self.cache, self.cache_size, self.cache_cap = OrderedDict(), 0, cache_bytes
        self.fetch_limit = fetch_limit
        self.final = OrderedDict()   # final URL string -> model tokens (str hash is cached on the object: cheap lookups)

    def _get(self, key):
        v = self.cache.get(key)
        if v is not None:
            self.cache.move_to_end(key)
        return v

    def _put(self, key, v):
        self.cache[key] = v
        self.cache_size += len(v[0] or "")
        while self.cache_size > self.cache_cap and self.cache:
            _, old = self.cache.popitem(last=False)
            self.cache_size -= len(old[0] or "")

    async def _bytes_of(self, url, session):
        if url.startswith("data:"):
            head, _, b64 = url.partition(",")
            if ";base64" not in head:
                return None
            return base64.b64decode(b64, validate=True)
        if url.startswith(("http://", "https://")) and session is not None:
            import aiohttp
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as r:
                if r.status != 200:
                    return None
                data = await r.content.read(self.fetch_limit + 1)
                return data if len(data) <= self.fetch_limit else None
        return None

    async def _one(self, part, session):
        """Shrinks one image part in place; returns (url chars, model tokens) for the router's length estimate."""
        iu = part.get("image_url")
        url = iu.get("url") if isinstance(iu, dict) else iu
        if not isinstance(url, str):
            return 0, 0
        key = hashlib.blake2b(url.encode("utf-8", "surrogatepass"), digest_size=16).hexdigest()
        hit = self._get(key)
        if hit is None:
            try:
                data = await self._bytes_of(url, session)
            except (binascii.Error, ValueError, OSError, asyncio.TimeoutError) as e:
                self.log.info(f"img: fetch/decode failed ({type(e).__name__}: {e}); passed through")
                data = None
            except Exception as e:
                self.log.info(f"img: fetch failed ({type(e).__name__}: {e}); passed through")
                data = None
            if data is None:
                return len(url), 0
            try:
                png, w, h = await asyncio.get_running_loop().run_in_executor(self.pool, _shrink, data, self.max_pixels)
            except Exception as e:
                self.log.info(f"img: shrink failed ({type(e).__name__}: {e}); passed through")
                return len(url), 0
            new = "data:image/png;base64," + base64.b64encode(png).decode() if png is not None else None
            if new is not None:
                self.log.info(f"img: {len(data)} B -> {w}x{h} PNG {len(png)} B")
            elif not url.startswith("data:"):
                new = None   # small remote image: vLLM fetches it itself
            hit = (new, w, h)
            self._put(key, hit)
        new, w, h = hit
        if new is not None:
            if isinstance(iu, dict):
                iu["url"] = new
            else:
                part["image_url"] = new
            url = new
        tok = (w // FACTOR) * (h // FACTOR) + 2
        self.final[url] = tok
        self.final.move_to_end(url)
        while len(self.final) > 512:
            self.final.popitem(last=False)
        return len(url), tok

    async def apply(self, body, session):
        """In place over body["messages"]; returns (chars of image URLs, model tokens of the images)."""
        if self.pool is None or "messages" not in body:
            return 0, 0
        parts = [p for m in body["messages"] if isinstance(m.get("content"), list)
                 for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url"]
        if not parts:
            return 0, 0
        res = await asyncio.gather(*(self._one(p, session) for p in parts))
        return sum(c for c, _ in res), sum(t for _, t in res)

    def measure(self, body):
        """(chars of image URLs, model tokens of the images) in body as it is now (e.g. after a compaction)."""
        chars = tok = 0
        for m in body.get("messages", []):
            c = m.get("content")
            if not isinstance(c, list):
                continue
            for p in c:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    iu = p.get("image_url")
                    url = iu.get("url") if isinstance(iu, dict) else iu
                    if isinstance(url, str):
                        chars += len(url)
                        tok += self.final.get(url, 0)
        return chars, tok
