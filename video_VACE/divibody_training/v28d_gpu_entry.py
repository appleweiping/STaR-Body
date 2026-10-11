"""Set a CUDA allocation cap before importing the copied training entry."""
import os
import runpy
import sys
import torch


def main():
    module = sys.argv[1]
    sys.argv = [module, *sys.argv[2:]]
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        budget = float(os.environ.get('V28D_MEMORY_LIMIT_GIB', '10')) * 1024**3
        fraction = min(1., budget / torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    runpy.run_module(module, run_name='__main__')


if __name__ == '__main__':
    main()
