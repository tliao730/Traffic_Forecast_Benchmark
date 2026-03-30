from typing import Callable, TypeVar

import torch

T = TypeVar("T")


def run_with_batch_size_backoff(
    run_once: Callable[[int], T],
    initial_batch_size: int,
    *,
    min_batch_size: int = 1,
) -> tuple[T, int]:
    """
    Retry a batched inference function by halving the batch size on CUDA OOM.

    Returns both the successful result and the final batch size used.
    """
    batch_size = initial_batch_size
    while True:
        try:
            return run_once(batch_size), batch_size
        except torch.cuda.OutOfMemoryError as exc:
            next_batch_size = batch_size // 2
            print(
                f"OutOfMemoryError at batch_size {batch_size}, reducing to {next_batch_size}"
            )
            if next_batch_size < min_batch_size:
                raise RuntimeError("batch_size reduced below 1, still OOM.") from exc
            batch_size = next_batch_size
