"""Registry of frozen image encoders + one uniform way to turn a JPEG into a vector.

ทุกตัวในนี้เป็น **zero-shot** ล้วน ๆ — โหลด weight ที่ pretrain มาแล้วใช้เป็น feature extractor
ไม่มีการ fine-tune อะไรทั้งสิ้น ต่างกันแค่ว่า pretrain มายังไง:

  ตระกูล CLIP / SigLIP  เทรนด้วยคู่ (ภาพ, ข้อความ) → เก่งเรื่อง "อ่าน" ตัวอักษรในโลโก้
  ตระกูล DINO           เทรนแบบ self-supervised ล้วน → เก่งเรื่องจับว่าเป็น "ชิ้นเดียวกัน"
                        (instance retrieval) แม้มุมกล้อง/แสง/ขนาดต่างกัน

โลโก้กินทั้งสองทาง (wordmark ต้องอ่าน, สัญลักษณ์ต้องจำรูปทรง) — benchmark เลยเป็นตัวตัดสิน
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger("img_search.encoders")


class NonFiniteEmbedding(RuntimeError):
    """Raised when a backbone overflows its inference precision and emits NaN/inf."""


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    kind: str          # hf-clip | hf-clip-pooler | hf-siglip | hf-dino | open-clip
    hub_id: str
    note: str = ""
    gated: bool = False


# ชื่อย่อ -> โมเดลจริง. ใช้ชื่อย่อได้ทุกที่ใน config (embed.models, benchmark.models)
ENCODERS: dict[str, EncoderSpec] = {
    # --- CLIP ---
    "clip-b32": EncoderSpec("clip-b32", "hf-clip", "openai/clip-vit-base-patch32",
                            "baseline ของโจทย์ (zeroshot_clip.py)"),
    "clip-l14": EncoderSpec("clip-l14", "hf-clip", "openai/clip-vit-large-patch14",
                            "ตัวที่ notebook รอบเดิมใช้"),
    "clip-l14-336": EncoderSpec("clip-l14-336", "hf-clip", "openai/clip-vit-large-patch14-336"),
    "clip-l14-pooler": EncoderSpec(
        "clip-l14-pooler", "hf-clip-pooler", "openai/clip-vit-large-patch14",
        "pooler_output ดิบของ vision tower — สิ่งที่ notebook รอบเดิมใช้จริง ไม่ใช่สเปซที่ CLIP "
        "เทรนมาให้วัดความคล้าย"),
    "clip-h14-laion": EncoderSpec("clip-h14-laion", "open-clip",
                                  "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"),
    "clip-bigg-laion": EncoderSpec("clip-bigg-laion", "open-clip",
                                   "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                                   "2.5B params — กิน VRAM เยอะ"),
    "eva02-l14-336": EncoderSpec("eva02-l14-336", "open-clip",
                                 "hf-hub:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k"),
    # --- SigLIP ---
    "siglip-so400m": EncoderSpec("siglip-so400m", "hf-siglip", "google/siglip-so400m-patch14-384"),
    "siglip2-b16": EncoderSpec("siglip2-b16", "hf-siglip", "google/siglip2-base-patch16-384"),
    "siglip2-so400m": EncoderSpec("siglip2-so400m", "hf-siglip", "google/siglip2-so400m-patch14-384"),
    "siglip2-g-opt": EncoderSpec("siglip2-g-opt", "hf-siglip", "google/siglip2-giant-opt-patch16-384"),
    # --- DINO (self-supervised) ---
    "dinov2-b": EncoderSpec("dinov2-b", "hf-dino", "facebook/dinov2-base"),
    "dinov2-l": EncoderSpec("dinov2-l", "hf-dino", "facebook/dinov2-large"),
    "dinov2-g": EncoderSpec("dinov2-g", "hf-dino", "facebook/dinov2-giant"),
    "dinov3-b": EncoderSpec("dinov3-b", "hf-dino", "facebook/dinov3-vitb16-pretrain-lvd1689m",
                            "ต้องกด accept license บน HF ก่อน", gated=True),
    "dinov3-l": EncoderSpec("dinov3-l", "hf-dino", "facebook/dinov3-vitl16-pretrain-lvd1689m",
                            "ต้องกด accept license บน HF ก่อน", gated=True),
}


def spec_for(key: str) -> EncoderSpec:
    if key in ENCODERS:
        return ENCODERS[key]
    # ยังเรียก hub id ตรง ๆ ได้ ถ้าอยากลองตัวที่ไม่ได้อยู่ใน registry
    if "/" in key:
        kind = "open-clip" if key.startswith("hf-hub:") else _guess_kind(key)
        return EncoderSpec(key, kind, key, "ad-hoc (ไม่อยู่ใน registry)")
    raise KeyError(f"unknown encoder {key!r}. known: {sorted(ENCODERS)}")


def _guess_kind(hub_id: str) -> str:
    low = hub_id.lower()
    if "siglip" in low:
        return "hf-siglip"
    if "dino" in low:
        return "hf-dino"
    return "hf-clip"


# --------------------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------------------

def make_view(img: Image.Image, view: str, size: int, pad_color: str = "edge") -> Image.Image:
    """Fit a PIL image into a `size x size` square.

    โลโก้ในภาพจริงมักอยู่ริมขอบ — `crop` (ค่า default ของ processor ทุกตัว) จึงตัดของสำคัญทิ้ง
    ได้ง่าย `pad` เป็นค่าที่ปลอดภัยกว่า เพราะไม่ตัดและไม่บิดสัดส่วน
    """
    img = img.convert("RGB")
    if view == "squash":
        return img.resize((size, size), Image.BICUBIC)
    if view == "crop":
        w, h = img.size
        scale = size / min(w, h)
        img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BICUBIC)
        w, h = img.size
        left, top = (w - size) // 2, (h - size) // 2
        return img.crop((left, top, left + size, top + size))
    if view == "pad":
        w, h = img.size
        side = max(w, h)
        if pad_color == "edge":
            fill = _edge_color(img)
        else:
            fill = (255, 255, 255) if pad_color == "white" else (0, 0, 0)
        canvas = Image.new("RGB", (side, side), fill)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas.resize((size, size), Image.BICUBIC)
    raise ValueError(f"unknown view {view!r} (expected pad | squash | crop)")


def _edge_color(img: Image.Image) -> tuple[int, int, int]:
    """Median colour of the 1-pixel border — โลโก้ส่วนใหญ่มีพื้นหลังเรียบ เติมสีนี้แล้วเนียน."""
    a = np.asarray(img)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]], axis=0)
    return tuple(int(v) for v in np.median(border, axis=0))


class _ViewDataset(Dataset):
    def __init__(self, paths: list[str], views: list[str], size: int,
                 mean: tuple, std: tuple, pad_color: str):
        self.paths, self.views, self.size, self.pad_color = paths, views, size, pad_color
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.paths[i]) as im:
            im.load()
            tiles = [make_view(im, v, self.size, self.pad_color) for v in self.views]
        out = torch.stack([torch.from_numpy(np.asarray(t).copy()) for t in tiles])  # V,H,W,3
        out = out.permute(0, 3, 1, 2).float().div_(255.0)
        return (out - self.mean) / self.std


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------

@dataclass
class LoadedEncoder:
    spec: EncoderSpec
    forward: Callable[[torch.Tensor], torch.Tensor]
    image_size: int
    mean: tuple
    std: tuple
    dim: int


def _from_pretrained(cls, hub_id: str, dtype):
    """transformers ย้ายจาก torch_dtype= ไป dtype= ระหว่างทาง — รองรับทั้งสองแบบ."""
    try:
        return cls.from_pretrained(hub_id, dtype=dtype)
    except TypeError:
        return cls.from_pretrained(hub_id, torch_dtype=dtype)


def _as_tensor(out) -> torch.Tensor:
    """transformers >=5 คืน `get_image_features` เป็น output object ที่เอา projected embedding
    ไปใส่ไว้ใน `pooler_output` — เวอร์ชันเก่าคืน tensor ตรง ๆ รองรับทั้งสองแบบ."""
    if torch.is_tensor(out):
        return out
    for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
        v = getattr(out, attr, None)
        if v is not None:
            return v if v.ndim == 2 else v.mean(dim=1)
    raise TypeError(f"cannot read an embedding out of {type(out).__name__}")


def _processor_geometry(hub_id: str, fallback: int) -> tuple[int, tuple, tuple]:
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(hub_id)
    size = getattr(proc, "crop_size", None) or getattr(proc, "size", None) or {}
    px = size.get("height") or size.get("shortest_edge") or fallback
    mean = tuple(getattr(proc, "image_mean", (0.5, 0.5, 0.5)))
    std = tuple(getattr(proc, "image_std", (0.5, 0.5, 0.5)))
    return int(px), mean, std


def inference_dtype(device: torch.device, half: bool = True) -> torch.dtype:
    """bfloat16 ถ้าการ์ดรองรับ ไม่งั้น float16.

    DINOv3 ตัวใหญ่มี activation ที่ทะลุช่วงของ float16 แล้วกลายเป็น NaN ทั้งก้อน
    bfloat16 กินหน่วยความจำเท่ากันแต่มีช่วง exponent เท่า float32 จึงไม่ล้น
    """
    if not half or device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_encoder(key: str, device: torch.device, dino_feature: str = "cls+avg",
                 half: bool = True) -> LoadedEncoder:
    spec = spec_for(key)
    dtype = inference_dtype(device, half)
    log.info("loading %s (%s, %s)", spec.key, spec.hub_id, dtype)

    if spec.kind == "open-clip":
        return _load_open_clip(spec, device, dtype)
    if spec.kind == "hf-dino":
        return _load_dino(spec, device, dtype, dino_feature)
    return _load_clip_like(spec, device, dtype, projected=spec.kind != "hf-clip-pooler")


def _load_clip_like(spec: EncoderSpec, device, dtype, projected: bool = True) -> LoadedEncoder:
    from transformers import AutoModel

    model = _from_pretrained(AutoModel, spec.hub_id, dtype).to(device).eval()
    px, mean, std = _processor_geometry(spec.hub_id, 224)

    def forward(x: torch.Tensor) -> torch.Tensor:
        if projected:
            return _as_tensor(model.get_image_features(pixel_values=x.to(dtype)))
        # ก่อน projection head — เอาไว้ทำซ้ำ baseline เดิมเพื่อเทียบเท่านั้น
        return model.vision_model(pixel_values=x.to(dtype)).pooler_output

    dim = _probe_dim(forward, px, device, dtype)
    return LoadedEncoder(spec, forward, px, mean, std, dim)


def _load_dino(spec: EncoderSpec, device, dtype, dino_feature: str) -> LoadedEncoder:
    from transformers import AutoModel

    model = _from_pretrained(AutoModel, spec.hub_id, dtype).to(device).eval()
    px, mean, std = _processor_geometry(spec.hub_id, 224)
    # DINOv3 มี register token คั่นระหว่าง CLS กับ patch token — ต้องข้ามให้ถูก
    n_prefix = 1 + int(getattr(model.config, "num_register_tokens", 0) or 0)

    def forward(x: torch.Tensor) -> torch.Tensor:
        h = model(pixel_values=x.to(dtype)).last_hidden_state
        cls = h[:, 0]
        if dino_feature == "cls":
            return cls
        # CLS + ค่าเฉลี่ยของ patch token คือ feature มาตรฐานที่ DINOv2 ใช้ตอน linear probe
        return torch.cat([cls, h[:, n_prefix:].mean(dim=1)], dim=-1)

    return LoadedEncoder(spec, forward, px, mean, std, _probe_dim(forward, px, device, dtype))


def _load_open_clip(spec: EncoderSpec, device, dtype) -> LoadedEncoder:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(spec.hub_id)
    model = model.to(device=device, dtype=dtype).eval()

    px, mean, std = 224, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    for t in getattr(preprocess, "transforms", []):
        name = type(t).__name__
        if name == "Normalize":
            mean, std = tuple(t.mean), tuple(t.std)
        elif name in ("Resize", "CenterCrop"):
            s = t.size
            px = int(s[0] if isinstance(s, (tuple, list)) else s)

    def forward(x: torch.Tensor) -> torch.Tensor:
        return model.encode_image(x.to(dtype))

    return LoadedEncoder(spec, forward, px, mean, std, _probe_dim(forward, px, device, dtype))


@torch.no_grad()
def _probe_dim(forward, px: int, device, dtype) -> int:
    return int(forward(torch.zeros(1, 3, px, px, device=device, dtype=dtype)).shape[-1])


# --------------------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------------------

@torch.no_grad()
def embed_paths(enc: LoadedEncoder, paths: list[str], *, views: list[str], pad_color: str,
                batch_size: int, num_workers: int, device: torch.device,
                progress: bool = True) -> np.ndarray:
    """Embed every path and return an L2-normalised float32 array of shape (N, dim).

    หลาย view ต่อหนึ่งภาพจะถูก normalise ทีละ view แล้วเฉลี่ยกัน (ensemble ระดับภาพ)
    """
    ds = _ViewDataset(paths, views, enc.image_size, enc.mean, enc.std, pad_color)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")

    iterator = loader
    if progress:
        import sys

        from tqdm import tqdm

        # ปิด progress bar เมื่อ stderr ไม่ใช่ terminal ไม่งั้น log ที่ redirect ลงไฟล์จะเละ
        iterator = tqdm(loader, desc=f"embed {enc.spec.key}", unit="batch",
                        disable=not sys.stderr.isatty())

    chunks: list[np.ndarray] = []
    for batch in iterator:                                   # B,V,3,H,W
        b, v = batch.shape[:2]
        flat = batch.flatten(0, 1).to(device, non_blocking=True)
        feats = enc.forward(flat).float()
        if not torch.isfinite(feats).all():
            raise NonFiniteEmbedding(
                f"{enc.spec.key} produced non-finite features in {flat.dtype} — "
                f"the activations overflowed this precision"
            )
        feats = torch.nn.functional.normalize(feats, dim=-1).view(b, v, -1).mean(dim=1)
        chunks.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)
