"""Export classifier r2plus1d (.pt) → ONNX → TensorRT engine FP16.

Bước trên đường lên edge: thay torch FP32 bằng engine FP16 (nhanh hơn, nhẹ hơn, đúng
thứ sẽ chạy trên Jetson). Giữ I/O fp32 để preprocess không đổi; FP16 nằm trong layer.

  python -m train.export_trt --ckpt weights/vio_r2plus1d.pt --out weights/vio_r2plus1d_fp16.engine
"""
from __future__ import annotations

import argparse
import os

import torch

from transport.model import build_backbone


def export_onnx(ckpt: str, onnx_path: str, clip_len: int, size: int,
                half: bool = True, opset: int = 17) -> None:
    ck = torch.load(ckpt, map_location="cpu")
    backbone = ck.get("backbone", "r2plus1d_18")
    model = build_backbone(backbone, num_classes=2, pretrained=False)
    model.load_state_dict(ck.get("model", ck))
    model.eval().cuda()
    dtype = torch.float16 if half else torch.float32
    if half:
        model = model.half()  # ONNX mang kiểu fp16 → engine strongly-typed chạy FP16
    dummy = torch.randn(1, 3, clip_len, size, size, device="cuda", dtype=dtype)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["clip"], output_names=["logits"],
        opset_version=opset, dynamo=False,
    )
    print(f"[onnx] {onnx_path}  backbone={backbone}  in=[1,3,{clip_len},{size},{size}] "
          f"dtype={dtype} opset={opset}")


def build_engine(onnx_path: str, engine_path: str) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TRT 11 bỏ BuilderFlag.FP16 → dùng STRONGLY_TYPED: precision theo kiểu trong ONNX (fp16).
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  ONNX parse error:", parser.get_error(i))
            raise SystemExit("parse ONNX thất bại")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    print("[trt] strongly-typed (FP16 theo ONNX)")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("build engine thất bại")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[trt] engine -> {engine_path}  ({os.path.getsize(engine_path)/1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights/vio_r2plus1d.pt")
    ap.add_argument("--onnx", default="weights/vio_r2plus1d.onnx")
    ap.add_argument("--out", default="weights/vio_r2plus1d_fp16.engine")
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-half", action="store_true", help="export FP32 (thử khi FP16 ONNX/TRT lỗi)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    clip_len = ck.get("clip_len", args.clip_len)
    size = ck.get("size", args.size)
    export_onnx(args.ckpt, args.onnx, clip_len, size, half=not args.no_half, opset=args.opset)
    build_engine(args.onnx, args.out)
    print("done.")


if __name__ == "__main__":
    main()
