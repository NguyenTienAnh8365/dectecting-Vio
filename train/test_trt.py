"""Test engine TensorRT FP16 của classifier: so prob + latency với torch FP32.

Chạy engine trên clip Violence/NonViolence thật, đối chiếu xác suất với model FP32 gốc
để chắc FP16 KHÔNG mất chất lượng, và đo độ trễ.

  python -m train.test_trt --engine weights/vio_r2plus1d_fp16.engine --n 8
"""
from __future__ import annotations

import argparse
import glob
import time

import numpy as np
import torch

from transport import config
from transport.model import VioModel
from train.train_clip import sample_frames


class TrtRunner:
    """Bọc engine TRT 11 (strongly-typed FP16). I/O qua torch tensor data_ptr (không pycuda)."""
    def __init__(self, engine_path: str):
        import tensorrt as trt
        self.trt = trt
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.in_name, self.out_name = None, None
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.in_name = n
            else:
                self.out_name = n
        self.in_dtype = self._np_dtype(self.engine.get_tensor_dtype(self.in_name))
        self.out_shape = tuple(self.engine.get_tensor_shape(self.out_name))
        self.out_dtype = self._np_dtype(self.engine.get_tensor_dtype(self.out_name))
        self.stream = torch.cuda.Stream()

    def _np_dtype(self, t):
        return {self.trt.float32: torch.float32, self.trt.float16: torch.float16}[t]

    def infer(self, x: np.ndarray) -> np.ndarray:
        """x: [1,3,T,H,W] fp32 numpy → logits numpy."""
        inp = torch.from_numpy(x).to("cuda", self.in_dtype).contiguous()
        out = torch.empty(self.out_shape, dtype=self.out_dtype, device="cuda")
        self.ctx.set_tensor_address(self.in_name, inp.data_ptr())
        self.ctx.set_tensor_address(self.out_name, out.data_ptr())
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return out.float().cpu().numpy()


def softmax_viol(logits: np.ndarray) -> float:
    e = np.exp(logits[0] - logits[0].max())
    return float((e / e.sum())[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="weights/vio_r2plus1d_fp16.engine")
    ap.add_argument("--ckpt", default="weights/vio_r2plus1d.pt")
    ap.add_argument("--data", default="data/RLVS/Real_Life_Violence_Dataset")
    ap.add_argument("--n", type=int, default=8, help="số clip mỗi lớp")
    args = ap.parse_args()

    size, clip_len = config.CLIP_SIZE, config.WIN_LEN
    fp32 = VioModel(ckpt=args.ckpt, half=False)         # baseline torch FP32
    trt_run = TrtRunner(args.engine)
    print(f"engine in_dtype={trt_run.in_dtype} out_shape={trt_run.out_shape}\n")

    viol = sorted(glob.glob(f"{args.data}/Violence/*.mp4"))[:args.n]
    nonv = sorted(glob.glob(f"{args.data}/NonViolence/*.mp4"))[:args.n]
    samples = [(p, 1) for p in viol] + [(p, 0) for p in nonv]

    print(f"{'clip':<22}{'label':<7}{'fp32':<8}{'trt16':<8}{'Δ':<8}")
    diffs, ok = [], 0
    for path, label in samples:
        arr = sample_frames(path, clip_len, size, train=False)  # [T,H,W,C]
        if arr is None:
            continue
        x = np.transpose(arr, (3, 0, 1, 2))[None].astype(np.float32)  # [1,C,T,H,W]
        with torch.no_grad():
            t = torch.from_numpy(x).cuda()
            p32 = torch.softmax(fp32.model(t).float(), 1)[0, 1].item()
        p16 = softmax_viol(trt_run.infer(x))
        d = abs(p32 - p16)
        diffs.append(d)
        ok += int((p32 >= 0.5) == (p16 >= 0.5))
        name = path.rsplit("/", 1)[-1]
        print(f"{name:<22}{label:<7}{p32:<8.3f}{p16:<8.3f}{d:<8.4f}")

    # Latency (chỉ tính infer, warmup 5)
    xw = np.transpose(sample_frames(viol[0], clip_len, size, False), (3, 0, 1, 2))[None].astype(np.float32)
    tin = torch.from_numpy(xw).cuda()
    for _ in range(5):
        with torch.no_grad():
            fp32.model(tin)
        trt_run.infer(xw)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        with torch.no_grad():
            fp32.model(tin)
    torch.cuda.synchronize()
    lat32 = (time.time() - t0) / 50 * 1000
    t0 = time.time()
    for _ in range(50):
        trt_run.infer(xw)
    torch.cuda.synchronize()
    lat16 = (time.time() - t0) / 50 * 1000

    n = len(diffs)
    print(f"\n== Kết quả ({n} clip) ==")
    print(f"  Khớp nhãn fp32 vs trt16: {ok}/{n}")
    print(f"  |Δprob|  trung bình={np.mean(diffs):.4f}  max={np.max(diffs):.4f}")
    print(f"  Latency  FP32(torch)={lat32:.2f} ms   FP16(TRT)={lat16:.2f} ms   speedup={lat32/lat16:.2f}x")


if __name__ == "__main__":
    main()
