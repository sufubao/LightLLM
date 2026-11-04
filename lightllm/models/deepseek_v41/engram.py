# MIT License
#
# Copyright (c) 2023 DeepSeek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Adapted from deepseek-ai/DeepSeek-V4.1-Flash, revision 314113f8a7dadb388215396ba5fbee2c610a04d6.

from dataclasses import dataclass
import numpy as np
import torch
from sympy import isprime


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    candidate = start + 1
    while not isprime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:

            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, tokenizer_vocab_size: int
) -> torch.Tensor:
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    n_heads: int
    head_dim: int

    @classmethod
    def from_args(cls, args) -> "EngramLayout | None":
        layer_ids = tuple(args.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size, n_heads = args.engram_max_ngram_size, args.engram_n_heads
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], args.engram_vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(args.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=args.engram_head_dim,
        )


class EngramHashes:
    def __init__(self, config, tokenizer, capacity, device):
        self.layout = EngramLayout.from_args(config)
        mapping, size = build_compressed_token_map(tokenizer)
        if size != config.engram_compressed_vocab_size:
            raise ValueError(
                f"Engram tokenizer normalization mismatch: {size}, expected {config.engram_compressed_vocab_size}"
            )
        self.pad = mapping[config.engram_pad_id]
        self.mapping = torch.tensor(mapping, device=device)
        self.multipliers = compute_hash_multipliers(self.layout.layer_ids, self.layout.max_ngram_size, size).to(device)
        self.primes = torch.tensor(self.layout.primes, device=device)
        flat = self.primes.flatten(1)
        self.offsets = flat.cumsum(-1) - flat
        self.cache = torch.full((capacity, self.layout.max_ngram_size - 1), self.pad, device=device, dtype=torch.long)

    def __call__(self, tokens, requests):
        output = []
        lookback = self.layout.max_ngram_size - 1
        for req, start, count, offset in requests:
            if start == 0:
                self.cache[req].fill_(self.pad)
            current = self.mapping[tokens[offset : offset + count]]
            history = torch.cat((self.cache[req], current))
            shifted = torch.stack(
                [history[lookback - shift : lookback - shift + count] for shift in range(lookback + 1)], -1
            )
            product = shifted[:, None, :] * self.multipliers
            rolling, hashes = product[..., 0], []
            for i in range(1, lookback + 1):
                rolling = torch.bitwise_xor(rolling, product[..., i])
                hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
            output.append(torch.cat(hashes, -1) + self.offsets)
            self.cache[req].copy_(history[-lookback:])
        return torch.cat(output)
