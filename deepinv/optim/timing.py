import torch
import timeit


class FixedPointTimer:
    def __init__(self, device):
        dtype = device.type
        self.device_type = dtype
        self.is_cpu = isinstance(dtype, str) and len(dtype) > 2 and ('cpu' in dtype)
        self.is_cuda = (not self.is_cpu) and ('cuda' in dtype)

        self.start_evt = None
        self.end_evt = None
        self.t0 = None  # used if cpu
        if self.is_cuda:
            self.start_evt = torch.cuda.Event(enable_timing=True)
            self.end_evt = torch.cuda.Event(enable_timing=True)

    def warmup(self):
        if self.is_cuda:
            self.start_evt.record()
            self.end_evt.record()
            torch.cuda.synchronize()
            self.start_evt.elapsed_time(self.end_evt)

    def start(self):
        if self.is_cuda:
            self.start_evt.record()
        elif self.is_cpu:
            self.t0 = timeit.default_timer()
        else:
            raise Exception(f'Unsupported device type {self.device_type}')
    def end(self):
        if self.is_cuda:
            self.end_evt.record()
            torch.cuda.synchronize()
            it_time = self.start_evt.elapsed_time(self.end_evt)  # milliseconds
        elif self.is_cpu:
            it_time = timeit.default_timer() - self.t0  # Time reported in seconds
            it_time = 1000 * it_time  # set same unit as for GPU
        else:
            raise Exception(f'Unsupported device type {self.device_type}')

        return it_time
