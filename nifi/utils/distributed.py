import os

import torch


def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    if is_distributed():
        return torch.distributed.get_rank()
    return 0


def world_size() -> int:
    if is_distributed():
        return torch.distributed.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        torch.distributed.barrier()


def maybe_init_distributed(backend: str = "nccl") -> None:
    if is_distributed():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        if world > 1:
            torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world)
