from lightllm.common.mtp_workspace import MTPWorkspaceAllocator


def test_workspace_allocator_retains_selected_owners_and_reuses_evicted_blocks():
    allocator = MTPWorkspaceAllocator(capacity=2)

    workspace, evicted, staged = allocator.prepare([7, 3])
    assert workspace == [0, 1]
    assert evicted == []
    assert staged == [(7, 0), (3, 1)]

    workspace, evicted, staged = allocator.prepare([3, 9])
    assert workspace == [1, 0]
    assert evicted == [(7, 0)]
    assert staged == [(9, 0)]


def test_workspace_allocator_release_returns_only_owned_requests():
    allocator = MTPWorkspaceAllocator(capacity=2)
    allocator.prepare([4, 5])

    assert allocator.release([5, 99]) == [(5, 1)]
    assert allocator.workspace_for(5) is None
    assert allocator.workspace_for(4) == 0


def test_workspace_allocator_defers_reuse_until_event_is_consumed():
    allocator = MTPWorkspaceAllocator(capacity=1)
    allocator.prepare([7])
    released = allocator.release([7])
    event = object()

    allocator.defer_reuse(released, event)
    assert allocator.take_reuse_event(0) is event
    assert allocator.take_reuse_event(0) is None
