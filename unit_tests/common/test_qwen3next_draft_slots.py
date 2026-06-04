def test_draft_layers_map_to_distinct_slots():
    # main full-att layers -> 0..M-1 ; draft layers -> M..M+D-1 (no overlap).
    M, D = 16, 2
    main_slots = set(range(M))
    draft_slots = {M + d for d in range(D)}
    assert main_slots.isdisjoint(draft_slots)
    assert max(draft_slots) == M + D - 1


def test_draft_kv_slot_mapping_via_interval_math():
    # Mirrors the runtime mapping in Qwen3_5MTPModel._assign_draft_kv_slot:
    # the shared Qwen3NextMemManager maps layer_index -> layer_index // full_attention_interval.
    # The draft sets layer_num_ = (main_full_att + draft_idx) * interval so the existing
    # `// interval` math lands the draft at a dedicated slot past all main slots.
    interval = 4
    main_full_att = 16  # n_layer=64, full_attention_interval=4 -> 16 main full-attn layers

    def mem_manager_slot(layer_index):
        return layer_index // interval

    # main full-attn layers 3,7,...,63 -> slots 0..15
    main_layers = [li for li in range(64) if (li + 1) % interval == 0]
    main_slots = {mem_manager_slot(li) for li in main_layers}
    assert main_slots == set(range(main_full_att))

    # draft layer with draft_idx=0 -> dedicated slot 16, non-colliding
    draft_idx = 0
    draft_layer_num_ = (main_full_att + draft_idx) * interval
    draft_slot = mem_manager_slot(draft_layer_num_)
    assert draft_slot == main_full_att + draft_idx == 16
    assert draft_slot not in main_slots
    assert main_full_att <= draft_slot < main_full_att + 1
