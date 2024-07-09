# mypy: ignore-errors
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import partial
from typing import Any, Tuple

from benchmark_runner import BenchmarkRunner  # type: ignore[import-not-found]

import torch

from torch._inductor.utils import fresh_inductor_cache
from torch.nn.attention.flex_attention import flex_attention


class BenchmarkRunnerFlexAttention(BenchmarkRunner):  # type: ignore[misc, no-any-unimported]
    def __init__(self) -> None:
        super().__init__("flex_attention")

    def create_input(self) -> Tuple[Any, ...]:
        batch_size = 2 ** random.randint(0, 9)
        num_heads = 16
        slen = 2 ** random.randint(7, 13)
        # head dims
        d = 64
        dtype = torch.float16
        device = torch.device("cuda")
        requires_grad = False
        print(batch_size, num_heads, slen, slen, d, dtype, device, requires_grad)
        return batch_size, num_heads, slen, slen, d, dtype, device, requires_grad

    def generate_inputs(
        self,
        batch_size: int,
        num_heads: int,
        q_sequence_length: int,
        kv_sequence_length: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        requires_grad: bool,
    ):
        q_shape = (batch_size, q_sequence_length, num_heads * head_dim)
        kv_shape = (batch_size, kv_sequence_length, num_heads * head_dim)

        make_q = partial(
            torch.rand, q_shape, device=device, dtype=dtype, requires_grad=requires_grad
        )
        make_kv = partial(
            torch.rand,
            kv_shape,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        query = (
            make_q()
            .view(batch_size, q_sequence_length, num_heads, head_dim)
            .transpose(1, 2)
        )
        key = (
            make_kv()
            .view(batch_size, kv_sequence_length, num_heads, head_dim)
            .transpose(1, 2)
        )
        value = (
            make_kv()
            .view(batch_size, kv_sequence_length, num_heads, head_dim)
            .transpose(1, 2)
        )
        return query, key, value

    def run_benchmark(
        self,
        batch_size: int,
        num_heads: int,
        q_sequence_length: int,
        kv_sequence_length: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        requires_grad: bool,
    ) -> Any:
        query, key, value = self.generate_inputs(
            batch_size,
            num_heads,
            q_sequence_length,
            kv_sequence_length,
            head_dim,
            dtype,
            device,
            requires_grad,
        )

        def noop(score, b, h, m, n):
            return score

        with fresh_inductor_cache():
            compiled_sdpa = torch.compile(
                flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
            )
            compiled_sdpa(query, key, value)
            torch.compiler.reset()


if __name__ == "__main__":
    runner = BenchmarkRunnerFlexAttention()
    runner.run()
