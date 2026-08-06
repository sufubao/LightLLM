from typing import Optional


RANDOM_SEED = -1
MAX_SEED = (1 << 63) - 1


def validate_seed(seed: int) -> None:
    if seed < RANDOM_SEED or seed > MAX_SEED:
        raise ValueError(f"seed must be {RANDOM_SEED} (random), or an integer in [0, {MAX_SEED}], got {seed}")


def normalize_seed(seed: Optional[int]) -> int:
    seed = RANDOM_SEED if seed is None else seed
    validate_seed(seed)
    return seed
