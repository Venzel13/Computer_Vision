import torch

from benchmark import Benchmark
from config import NETS
from save import save_benchmarks

N, C, H, W = 1, 3, 224, 224
image = torch.rand(N, C, H, W)
image = image.cuda()
benchmark = Benchmark()
stats = benchmark.compute_stats(image=image, constructors=NETS)
save_benchmarks(stats, filename="benchmarks.csv")