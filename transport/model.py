"""Violence classifier clip-level — bọc torchvision.models.video (chỉ phụ thuộc torch/
torchvision sẵn có, tránh pytorchvideo).

Backbone Kinetics-pretrained (r2plus1d_18 / mc3_18 / r3d_18), thay đầu fc → nhị phân
(0=non-violence, 1=violence). Fine-tune trên RLVS/RWF-2000 bằng train/train_clip.py.

Không có checkpoint → DUMMY mode: trả prob=0.0 (pipeline chạy thông để test transport,
KHÔNG bắn alert giả). Phải fine-tune rồi trỏ VIO_CKPT mới có giá trị thật.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from . import config

# Chuẩn hóa Kinetics (đúng tiền xử lý weights pretrained torchvision video).
_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

_BACKBONES = {"r2plus1d_18", "mc3_18", "r3d_18", "s3d", "mvit_v1_b", "mvit_v2_s"}


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
    elif hasattr(model, "head"):  # mvit
        model.head = nn.Linear(model.head[-1].in_features, num_classes)
    elif hasattr(model, "classifier"):  # s3d
        model.classifier[-1] = nn.Conv3d(model.classifier[-1].in_channels, num_classes, 1)
    return model


class VioModel:
    def __init__(self, ckpt: str | None = None, backbone: str | None = None,
                 device: str | None = None, half: bool | None = None):
        self.backbone = backbone or config.VIO_BACKBONE
        self.device = device or config.VIO_DEVICE
        self.half = config.VIO_HALF if half is None else half
        self.dummy = True
        self.model = None
        ckpt = ckpt if ckpt is not None else config.VIO_CKPT

        if ckpt and os.path.exists(ckpt):
            import torch
            self.torch = torch
            self.model = build_backbone(self.backbone, num_classes=2, pretrained=False)
            state = torch.load(ckpt, map_location="cpu")
            self.model.load_state_dict(state.get("model", state))
            dev = f"cuda:{self.device}" if self.device.isdigit() else self.device
            self.model = self.model.to(dev).eval()
            if self.half and dev != "cpu":
                self.model = self.model.half()
            self.dev = dev
            self.dummy = False
            print(f"[model] loaded {self.backbone} ckpt={ckpt} device={dev} half={self.half}")
        else:
            print(f"[model] ⚠ không thấy checkpoint ({ckpt}) → DUMMY mode (prob=0). "
                  f"Fine-tune bằng train/train_clip.py rồi trỏ VIO_CKPT.")

    def preprocess(self, frames: list[np.ndarray]):
        """list BGR (len = WIN_LEN) → tensor [1, C, T, H, W] chuẩn Kinetics."""
        size = config.CLIP_SIZE
        clip = []
        for f in frames:
            f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            f = (f - _MEAN) / _STD
            clip.append(f)
        arr = np.stack(clip, axis=0)            # [T, H, W, C]
        arr = np.transpose(arr, (3, 0, 1, 2))   # [C, T, H, W]
        t = self.torch.from_numpy(arr).unsqueeze(0)  # [1, C, T, H, W]
        t = t.to(self.dev)
        return t.half() if (self.half and self.dev != "cpu") else t

    def predict(self, frames: list[np.ndarray]) -> float:
        """Trả prob bạo lực [0..1] cho 1 clip."""
        if self.dummy or self.model is None:
            return 0.0
        with self.torch.no_grad():
            x = self.preprocess(frames)
            logits = self.model(x)
            prob = self.torch.softmax(logits.float(), dim=1)[0, 1].item()
        return float(prob)
