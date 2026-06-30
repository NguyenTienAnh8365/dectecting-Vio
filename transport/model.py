"""Violence classifier clip-level — bọc torchvision.models.video (chỉ phụ thuộc torch/
torchvision sẵn có, tránh pytorchvideo).

Backbone Kinetics-pretrained (r2plus1d_18 / mc3_18 / r3d_18), thay đầu fc → nhị phân
(0=non-violence, 1=violence). Fine-tune trên RLVS/RWF-2000 bằng train/train_clip.py.

Không có checkpoint → DUMMY mode: trả prob=0.0 (pipeline chạy thông để test transport,
KHÔNG bắn alert giả). Phải fine-tune rồi trỏ VIO_CKPT mới có giá trị thật.
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np

from . import config

# Spec tiền xử lý theo BACKBONE — nguồn chân lý DUY NHẤT cho size/clip_len/normalize, dùng
# chung cho train (train_clip.py) lẫn infer (preprocess) để KHÔNG bao giờ lệch train/serve.
# CNN video (r2plus1d/mc3/r3d): 112×112, mean/std torchvision VideoClassification.
# Transformer (mvit_*): 224×224, 16 frame CỐ ĐỊNH (positional-embed gắn cứng), norm Kinetics [0.45]/[0.225].
_KIN_CNN_MEAN, _KIN_CNN_STD = [0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]
_KIN_T_MEAN, _KIN_T_STD = [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
BACKBONE_SPEC = {
    "r2plus1d_18": dict(size=112, clip_len=16, mean=_KIN_CNN_MEAN, std=_KIN_CNN_STD),
    "mc3_18":      dict(size=112, clip_len=16, mean=_KIN_CNN_MEAN, std=_KIN_CNN_STD),
    "r3d_18":      dict(size=112, clip_len=16, mean=_KIN_CNN_MEAN, std=_KIN_CNN_STD),
    "s3d":         dict(size=224, clip_len=16, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    "mvit_v1_b":   dict(size=224, clip_len=16, mean=_KIN_T_MEAN, std=_KIN_T_STD),
    "mvit_v2_s":   dict(size=224, clip_len=16, mean=_KIN_T_MEAN, std=_KIN_T_STD),
}
_BACKBONES = set(BACKBONE_SPEC)


def spec_of(backbone: str) -> dict:
    if backbone not in BACKBONE_SPEC:
        raise ValueError(f"backbone không hỗ trợ: {backbone} (chọn {sorted(BACKBONE_SPEC)})")
    return BACKBONE_SPEC[backbone]


def even_indices(n: int, k: int) -> list[int]:
    """k chỉ số trải đều trong [0, n-1] (linspace). Dùng để lấy 16 frame đều từ buffer span thời gian
    → khớp cách training sample (16 frame trải đều cả clip), tránh lệch train/serve về thời gian."""
    if n <= 0:
        return [0] * k
    return [int(round(x)) for x in np.linspace(0, n - 1, k)]


def multispan_predict(model, frames: list, boxes: list, spans_frames: list[int], W: int, H: int):
    """MAX prob qua nhiều SPAN thời gian. frames/boxes = buffer dài (= span lớn nhất).

    Với mỗi span: lấy `span` frame gần nhất → even_indices WIN_LEN → crop union người CỦA các frame
    đó → predict. Lấy max. Bắt cả đánh nhau NHANH (span ngắn) lẫn CHẬM/đám đông (span dài) —
    1 span cố định không kham được cả hai (vd fight dashcam ~7s mù với span 3s). Trả (prob, crop_box).
    """
    n = len(frames)
    clips, cbs = [], []
    for sf in sorted(set(int(s) for s in spans_frames)):
        sf = min(max(sf, config.WIN_LEN), n)
        seg_f, seg_b = frames[n - sf:], boxes[n - sf:]
        sel = even_indices(len(seg_f), config.WIN_LEN)
        base = [seg_f[i] for i in sel]
        # Thử CẢ full-frame LẪN crop-người → lấy max. Crop CHỈ giúp (cận cảnh CCTV), không bao giờ
        # hại: cảnh đám đông/dashcam góc rộng thì full-frame mới bắt được (crop union ≈ cả khung).
        clips.append(base); cbs.append(None)
        if config.CROP_TO_PERSON:
            cb = person_crop_box([seg_b[i] for i in sel], W, H, config.CROP_PAD)
            if cb:
                x1, y1, x2, y2 = cb
                clips.append([f[y1:y2, x1:x2] for f in base]); cbs.append(cb)
    probs = model.predict_batch(clips)  # GỘP 1 forward batch (nhanh hơn N lần tuần tự)
    best, best_box = 0.0, None
    for p, cb in zip(probs, cbs):
        if p > best:
            best, best_box = p, cb
    return best, best_box


def frames_to_clip(frames: list[np.ndarray], backbone: str, *, flip: bool = False) -> np.ndarray:
    """list ảnh BGR → ndarray [C,T,H,W] float32 đã chuẩn hóa theo spec của backbone.

    DÙNG CHUNG train+infer: resize vuông (INTER_AREA) → BGR→RGB → /255 → (x-mean)/std → [C,T,H,W].
    `flip` (chỉ train) lật ngang nhất quán mọi frame trong clip. Đây là gốc rễ chống train/serve skew.
    """
    sp = spec_of(backbone)
    size = sp["size"]
    mean = np.asarray(sp["mean"], dtype=np.float32)
    std = np.asarray(sp["std"], dtype=np.float32)
    clip = []
    for f in frames:
        f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
        if flip:
            f = cv2.flip(f, 1)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        clip.append((f - mean) / std)
    arr = np.stack(clip, axis=0)            # [T, H, W, C]
    return np.transpose(arr, (3, 0, 1, 2))  # [C, T, H, W]


class _TrtRunner:
    """Chạy engine TensorRT (FP16, strongly-typed). I/O qua torch tensor data_ptr — không pycuda.

    Cùng pattern train/test_trt.py nhưng nằm trong transport để classifier live dùng được.
    """
    def __init__(self, engine_path: str, device: str = "cuda:0"):
        import tensorrt as trt
        import torch
        self.trt, self.torch, self.device = trt, torch, device
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.in_name = self.out_name = None
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.in_name = n
            else:
                self.out_name = n
        _np = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.in_dtype = _np[self.engine.get_tensor_dtype(self.in_name)]
        self.out_shape = tuple(self.engine.get_tensor_shape(self.out_name))
        self.out_dtype = _np[self.engine.get_tensor_dtype(self.out_name)]
        self.stream = torch.cuda.Stream()

    def infer(self, x):
        """x: torch tensor [1,C,T,H,W] trên cuda → prob lớp 1 (bạo lực)."""
        inp = x.to(self.device, self.in_dtype).contiguous()
        out = self.torch.empty(self.out_shape, dtype=self.out_dtype, device=self.device)
        self.ctx.set_tensor_address(self.in_name, inp.data_ptr())
        self.ctx.set_tensor_address(self.out_name, out.data_ptr())
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return self.torch.softmax(out.float(), dim=1)[0, 1].item()


def quiet_ml_logs() -> None:
    """Tắt log ồn của ultralytics (torchvision-incompat, guess-task, "Loading…") — qua logger Python."""
    import logging
    os.environ.setdefault("YOLO_VERBOSE", "False")
    logging.getLogger("ultralytics").setLevel(logging.ERROR)


import contextlib


@contextlib.contextmanager
def suppress_native_logs():
    """Nuốt log C/C++ (TensorRT [I]/[W], engine-size…) bằng cách tạm chuyển fd 1+2 sang /dev/null.

    Log của TRT in thẳng ở tầng native nên redirect_stdout (Python) không bắt được — phải đổi fd.
    Chỉ bọc lúc NẠP engine (1 lần lúc khởi động), không bọc lúc chạy để không nuốt log thật.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    old1, old2 = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old1, 1)
        os.dup2(old2, 2)
        for fd in (devnull, old1, old2):
            os.close(fd)


def person_crop_box(boxes_per_frame: list, W: int, H: int, pad: float):
    """Union mọi bbox người trong cửa sổ → hộp VUÔNG có đệm, kẹp trong khung.

    boxes_per_frame: list theo từng frame, mỗi phần tử là list bbox [x,y,w,h] (góc trên-trái).
    Trả (x1,y1,x2,y2) để crop, hoặc None nếu không có người / hộp quá nhỏ (→ caller dùng cả khung).
    Vuông hóa: hành động bạo lực thường lan theo cả 2 chiều; tránh méo khi resize 112×112.
    """
    pts = [b for bs in boxes_per_frame for b in bs]
    if not pts:
        return None
    x1 = min(b[0] for b in pts)
    y1 = min(b[1] for b in pts)
    x2 = max(b[0] + b[2] for b in pts)
    y2 = max(b[1] + b[3] for b in pts)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * pad)  # cạnh vuông + đệm 2 bên
    x1, x2 = int(cx - side / 2), int(cx + side / 2)
    y1, y2 = int(cy - side / 2), int(cy + side / 2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 16 or y2 - y1 < 16:  # hộp suy biến → để caller dùng cả khung
        return None
    return x1, y1, x2, y2


def build_backbone(name: str, num_classes: int = 2, pretrained: bool = True):
    """Tạo model torchvision video với đầu phân loại num_classes. Dùng chung cho train + infer."""
    import torch.nn as nn
    import torchvision.models.video as tvv

    if name not in _BACKBONES:
        raise ValueError(f"backbone không hỗ trợ: {name} (chọn {_BACKBONES})")
    weights = "KINETICS400_V1" if pretrained else None
    model = getattr(tvv, name)(weights=weights)
    # Thay đầu phân loại — vị trí khác nhau theo kiến trúc.
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "head"):  # mvit: head = Sequential(Dropout, Linear) → chỉ thay Linear, GIỮ Dropout
        model.head[-1] = nn.Linear(model.head[-1].in_features, num_classes)
    elif hasattr(model, "classifier"):  # s3d
        model.classifier[-1] = nn.Conv3d(model.classifier[-1].in_channels, num_classes, 1)
    return model
    

class VioModel:
    def __init__(self, ckpt: str | None = None, backbone: str | None = None,
                 device: str | None = None, half: bool | None = None,
                 engine: str | None = None):
        self.backbone = backbone or config.VIO_BACKBONE
        self.device = device or config.VIO_DEVICE
        self.half = config.VIO_HALF if half is None else half
        self.dummy = True
        self.mode = "dummy"
        self.model = None
        self.engine = None
        engine = engine if engine is not None else config.VIO_ENGINE
        ckpt = ckpt if ckpt is not None else config.VIO_CKPT
        dev = f"cuda:{self.device}" if self.device.isdigit() else self.device
        self.dev = dev

        if engine and os.path.exists(engine):
            # Ưu tiên engine TensorRT FP16 (full-TRT) — preprocess ra fp16, infer qua engine.
            import torch
            self.torch = torch
            self.engine = _TrtRunner(engine, device=dev if dev != "cpu" else "cuda:0")
            self.half = self.engine.in_dtype == torch.float16
            self.dummy = False
            self.mode = "engine"
            print(f"[model] loaded TensorRT engine={engine} device={dev} in_dtype={self.engine.in_dtype}")
        elif ckpt and os.path.exists(ckpt):
            import torch
            self.torch = torch
            state = torch.load(ckpt, map_location="cpu")
            # Ưu tiên backbone lưu trong checkpoint (tránh lệch nếu VIO_BACKBONE cấu hình sai).
            if isinstance(state, dict) and state.get("backbone"):
                self.backbone = state["backbone"]
            self.model = build_backbone(self.backbone, num_classes=2, pretrained=False)
            self.model.load_state_dict(state.get("model", state))
            self.model = self.model.to(dev).eval()
            if self.half and dev != "cpu":
                self.model = self.model.half()
            self.dummy = False
            self.mode = "torch"
            print(f"[model] loaded {self.backbone} ckpt={ckpt} device={dev} half={self.half}")
        else:
            print(f"[model] ⚠ không thấy engine/checkpoint → DUMMY mode (prob=0). "
                  f"Train rồi trỏ VIO_ENGINE (.engine) hoặc VIO_CKPT (.pt).")

    def preprocess(self, frames: list[np.ndarray]):
        """list BGR (len = WIN_LEN) → tensor [1, C, T, H, W] theo spec backbone (đồng nhất với train)."""
        arr = frames_to_clip(frames, self.backbone)  # [C, T, H, W]
        t = self.torch.from_numpy(arr).unsqueeze(0).to(self.dev)  # [1, C, T, H, W]
        return t.half() if (self.half and self.dev != "cpu") else t

    def predict(self, frames: list[np.ndarray]) -> float:
        """Trả prob bạo lực [0..1] cho 1 clip."""
        if self.dummy:
            return 0.0
        with self.torch.no_grad():
            x = self.preprocess(frames)
            if self.mode == "engine":
                prob = self.engine.infer(x)
            else:
                logits = self.model(x)
                prob = self.torch.softmax(logits.float(), dim=1)[0, 1].item()
        # Chặn NaN/inf (FP16 thi thoảng tràn) → state/JSON không vỡ.
        return float(prob) if math.isfinite(prob) else 0.0

    def predict_batch(self, clips: list[list[np.ndarray]]) -> list[float]:
        """Nhiều clip → 1 forward batch (GPU dùng hết công suất; batch-N ≈ batch-1 trên Blackwell).

        Dùng cho multi-span (gộp các biến thể span×crop thành 1 lần infer thay vì N lần tuần tự).
        Engine TRT batch tĩnh =1 → fallback loop predict.
        """
        if self.dummy:
            return [0.0] * len(clips)
        if not clips:
            return []
        if self.mode == "engine":
            return [self.predict(c) for c in clips]
        with self.torch.no_grad():
            xs = [self.torch.from_numpy(frames_to_clip(c, self.backbone)) for c in clips]
            x = self.torch.stack(xs).to(self.dev)
            if self.half and self.dev != "cpu":
                x = x.half()
            probs = self.torch.softmax(self.model(x).float(), dim=1)[:, 1].tolist()
        return [float(p) if math.isfinite(p) else 0.0 for p in probs]
