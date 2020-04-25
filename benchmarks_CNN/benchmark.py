import torch
from torchvision import models


class Benchmark(object):
    def __init__(self):
        pass

    def compute_parameter_num(self, model):
        n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return n_param

    def measure_time_memory(self, image, model, forward_time_only=False):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

        if forward_time_only:
            start.record()
            with torch.no_grad():
                model(image)
            end.record()
            torch.cuda.synchronize()
            time = start.elapsed_time(end)
        else:
            start.record()
            y = model(image)
            y = y.sum()
            y.backward()
            end.record()
            torch.cuda.synchronize()
            time = start.elapsed_time(end)

        allocated_max = torch.cuda.max_memory_allocated()

        return time, allocated_max

    def compute_stats(self, image, constructors):
        stats = []

        for name, constructor in constructors.items():
            model = constructor()
            param = self.compute_parameter_num(model)
            model = model.cuda()
            time, memory = self.measure_time_memory(image, model)
            time_forward, memory_forward = self.measure_time_memory(
                image, model, forward_time_only=True
            )

            key = name
            values = [name, param, time, time_forward, memory, memory_forward]
            stats.append(
                {
                    "Model name": name,
                    "Number of parameters": param,
                    "Full pass time": time,
                    "Forward time": time_forward,
                    "Full memory usage": memory,
                    "Forward memory usage": memory_forward,
                }
            )

        return stats
