from types import SimpleNamespace

import pytest

from lightllm.server.router.req_queue.chunked_prefill.impl_for_pd_decode import PDDecodeQueue


def req(input_tokens, remaining, cached=0, ready=True, dp=0):
    return SimpleNamespace(
        sample_params=SimpleNamespace(suggested_dp_index=dp),
        prompt_cache_len=cached,
        get_tuple_tokens=lambda *_: (input_tokens, remaining),
        is_infer_decode=lambda: ready,
    )


def estimate(reqs, referenced, cache=True, dp=0, big_page_tokens=10000000):
    q = PDDecodeQueue.__new__(PDDecodeQueue)
    q.dp_index = dp
    q.args = SimpleNamespace(
        disable_dynamic_prompt_cache=not cache,
        linear_att_hash_page_size=big_page_tokens,
        linear_att_page_block_num=1,
        max_req_total_len=1000000,
    )
    q.router = SimpleNamespace(
        router_statics=SimpleNamespace(ema_req_out_len=300),
        radix_cache_client=SimpleNamespace(get_refed_tokens_num=lambda rank: referenced),
    )
    q.is_busy = lambda: False
    return q._caclu_batch_estimated_peak_token_num(SimpleNamespace(reqs=reqs))


def test_shared_prefix_is_counted_once():
    assert estimate([req(1000, 10, 800), req(1000, 10, 800)], referenced=800) == 1220


def test_unrelated_cached_prefixes_do_not_reduce_the_logical_bound():
    assert estimate([req(1000, 10, 800), req(1000, 10, 800)], referenced=1600) == 2020


def test_long_running_request_keeps_the_shared_prefix_after_other_requests_exit():
    assert estimate([req(100, 1000, 100), req(100, 0, 100)], referenced=100) == 1100


def test_not_initialized_requests_reserve_the_full_prompt():
    assert estimate([req(1000, 10, 800), req(1000, 10, ready=False)], referenced=800) == 2020


def test_pending_transfer_reserves_unmatched_input_and_output():
    assert estimate([req(1000, 10, 800), req(1000, 10, 800, ready=False)], referenced=800) == 1220


def test_cache_disabled_keeps_the_original_estimate():
    assert estimate([req(1000, 10, 800), req(1000, 10, 800)], referenced=800, cache=False) == 2020


def test_other_dp_requests_are_excluded():
    assert estimate([req(1000, 10, 800), req(1000, 10, 800, dp=1)], referenced=800) == 1010


@pytest.mark.parametrize("ready", [False, True])
def test_output_and_page_margins_are_preserved(ready):
    assert estimate(
        [req(1000, 8192 + 25, 800, ready), req(1000, 8192 + 25, 800, ready)], referenced=800
    ) == 1200 + 2 * (8192 + 25)


def test_estimate_bounds_physical_occupancy_with_nested_shared_prefixes():
    import random

    rng = random.Random(11)
    for _ in range(1000):
        layout = [
            (rng.randrange(1, 30), rng.randrange(0, 30), rng.randrange(0, 10)) for _ in range(rng.randrange(1, 8))
        ]
        layout = [(max(a, c), b, c) for a, b, c in layout]
        referenced = max(c for a, b, c in layout)
        bound = estimate([req(a, b, c) for a, b, c in layout], referenced)
        # All cached prefixes are nested. At each completion boundary, count
        # their physical union and each surviving request's private allocation.
        for t in {0, *(b for a, b, c in layout)}:
            live = [(a, b, c) for a, b, c in layout if b >= t]
            physical = max((c for a, b, c in live), default=0) + sum(a - c + t for a, b, c in live)
            assert bound >= physical


def test_hybrid_big_pages_keep_the_original_capacity_bound():
    # Only 80 of the reported 90 cached tokens share physical storage. The
    # remaining ten tokens were copied when restoring a small-page checkpoint.
    assert estimate([req(100, 0, 90), req(100, 0, 90)], referenced=80, big_page_tokens=40) == 200


def test_estimate_bounds_disjoint_and_shared_prefix_groups():
    import random

    rng = random.Random(12)
    for _ in range(1000):
        layout = [
            (rng.randrange(1, 50), rng.randrange(0, 30), rng.randrange(0, 20), rng.randrange(4))
            for _ in range(rng.randrange(1, 8))
        ]
        layout = [(max(a, c), b, c, group) for a, b, c, group in layout]
        referenced = sum(max((c for a, b, c, g in layout if g == group), default=0) for group in range(4))
        bound = estimate([req(a, b, c) for a, b, c, group in layout], referenced)
        for t in {0, *(b for a, b, c, group in layout)}:
            live = [(a, b, c, group) for a, b, c, group in layout if b >= t]
            shared = sum(max((c for a, b, c, g in live if g == group), default=0) for group in range(4))
            assert bound >= shared + sum(a - c + t for a, b, c, group in live)
