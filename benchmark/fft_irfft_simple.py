import sys
sys.path.insert(0, 'src')

import torch

# Simple benchmark for fft_irfft
if __name__ == "__main__":
    # Use power of 2 sizes for half precision compatibility
    shapes = [(16,), (32,), (64,), (128,), (256,), (16, 16), (32, 32)]

    print("FFT irfft Benchmark")
    print("=" * 60)

    # Only test float32 for now since half precision has restrictions
    dtype = torch.float32
    print(f"\ndtype = {dtype}")
    print("-" * 40)

    for shape in shapes:
        n_fft = shape[-1]
        input_shape = list(shape)
        input_shape[-1] = n_fft // 2 + 1

        # Create real input and convert to half-Hermitian
        real_input = torch.randn(input_shape, dtype=dtype, device='cuda')
        inp = torch.fft.rfft(real_input, dim=-1)

        # Warmup
        for _ in range(10):
            _ = torch.fft.irfft(inp, n=n_fft, dim=-1)

        # Benchmark
        import time
        N = 100
        start = time.time()
        for _ in range(N):
            _ = torch.fft.irfft(inp, n=n_fft, dim=-1)
        end = time.time()

        latency_ms = (end - start) / N * 1000
        print(f"  shape={shape}: {latency_ms:.4f} ms")

    print("\nDone!")