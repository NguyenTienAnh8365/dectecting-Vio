"""Export classifier (.pt) → ONNX → TensorRT engine FP16 (hỗ trợ CẢ mvit lẫn r2plus1d).

Bước trên đường lên edge: engine FP16 đúng thứ sẽ chạy trên Jetson. Giữ I/O fp16 (strongly-typed).

LỊCH SỬ + KẾT QUẢ ĐO (2026-07-02, RTX 3060 12GB ≈ A2000 12GB):
  • Exporter TorchScript cũ (dynamo=False) sinh op SequenceAt/Empty/Insert từ pooling-attention
    của MViT → TRT parser 11.x fail. DYNAMO exporter (torch ≥2.12) sinh đồ thị chuẩn → parse OK.
  • Parity torch vs TRT: max |Δprob| = 0.0003 trên input ngẫu nhiên — dùng được.
  • Tốc độ MViT trên Ampere: TRT batch-1 tĩnh ≈ torch (1.05x), batch-4 tĩnh CHẬM HƠN (0.80x),
    dynamic-batch CHẬM HẲN (0.5x — TRT chọn kernel kém khi shape động trên transformer).
    → Trên A2000/3060 GIỮ torch-FP16 làm mặc định (VIO_ENGINE để rỗng); engine chỉ đáng dùng
    trên Jetson (nơi TRT thường thắng rõ) — nhớ ĐO LẠI trên phần cứng đích trước khi bật.
  • Engine gắn chặt GPU build (sm khác nhau không chạy chéo) — build lại trên máy triển khai.

  python -m train.export_trt --ckpt weights/vio_mvit_v2_s.pt --out weights/vio_mvit_v2_s_fp16.engine
"""
from __future__ import annotations

import argparse
import os

import torch

from transport.model import build_backbone


def export_onnx(ckpt: str, onnx_path: str, clip_len: int, size: int,
                half: bool = True, batch: int = 1) -> None:
    ck = torch.load(ckpt, map_location="cpu")
    backbone = ck.get("backbone", "mvit_v2_s")
    model = build_backbone(backbone, num_classes=2, pretrained=False)
    model.load_state_dict(ck.get("model", ck))
    model.eval().cuda()
    dtype = torch.float16 if half else torch.float32
    if half:
        model = model.half()  # ONNX mang kiểu fp16 → engine strongly-typed chạy FP16
    dummy = torch.randn(batch, 3, clip_len, size, size, device="cuda", dtype=dtype)
    # dynamo=True BẮT BUỘC cho mvit (xem docstring); r2plus1d cũng chạy tốt qua dynamo.
    prog = torch.onnx.export(model, (dummy,), dynamo=True,
                             input_names=["clip"], output_names=["logits"])
    prog.optimize()
    prog.save(onnx_path)
    print(f"[onnx] {onnx_path}  backbone={backbone}  in=[{batch},3,{clip_len},{size},{size}] "
          f"dtype={dtype} (dynamo)")


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
    ap.add_argument("--ckpt", default="weights/vio_mvit_v2_s.pt")
    ap.add_argument("--onnx", default="weights/vio_mvit_v2_s_fp16.onnx")
    ap.add_argument("--out", default="weights/vio_mvit_v2_s_fp16.engine")
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=1,
                    help="batch TĨNH của engine (dynamic-batch đã đo: chậm, đừng dùng)")
    ap.add_argument("--no-half", action="store_true", help="export FP32 (thử khi FP16 ONNX/TRT lỗi)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    clip_len = ck.get("clip_len", args.clip_len)
    size = ck.get("size", args.size)
    export_onnx(args.ckpt, args.onnx, clip_len, size, half=not args.no_half, batch=args.batch)
    build_engine(args.onnx, args.out)
    print("done.")


if __name__ == "__main__":
    main()
