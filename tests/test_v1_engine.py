import tempfile
from types import SimpleNamespace
from typing import Any, Dict

try:
    import pytest  # type: ignore
except Exception:
    pytest = None

HAS_TORCH = True
try:
    import torch as _torch  # type: ignore
    torch: Any = _torch
except Exception:
    HAS_TORCH = False
    torch = object()  # type: ignore[assignment]

HAS_ETCD3 = True
try:
    import etcd3 as _etcd3  # type: ignore
    etcd3: Any = _etcd3
except Exception:
    HAS_ETCD3 = False
    etcd3 = object()  # type: ignore[assignment]

if pytest is not None and (not HAS_TORCH or not HAS_ETCD3):
    pytest.skip("Requires torch and etcd3 runtime; skipping v1 engine test.", allow_module_level=True)


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)


class _VLLMConfig:
    def __init__(self, storage_dir: str):
        # minimal kv_transfer_config used by engine
        self.kv_transfer_config = SimpleNamespace(
            storage_type="local",
            storage_dir=storage_dir,
            local_dir=storage_dir,
            etcd_endpoints=["127.0.0.1:2379"],
            enable_ssd_caching=False,
            ssd_cache_dir=storage_dir,
            enable_prefetch=False,
            chunk_size=8,
            force_sync_store=True,
            engine_id="v1_test_session",
        )
        # placeholders required by store_kv signature
        self.model_config = SimpleNamespace()
        self.parallel_config = SimpleNamespace()


class _Request:
    def __init__(self, request_id: str, tokens: list[int]):
        self.request_id = request_id
        self.prompt_token_ids = tokens
        self.num_tokens = len(tokens)
        self.num_computed_tokens = 0
        self.model = _FakeModel()


class _ForwardContext:
    def __init__(self, model: Any, kv_caches: Dict[str, Any]):
        self.model = model
        self.kv_caches = kv_caches
        # requests map is not strictly required here, engine keeps unfinished map


def _make_layer_kv(batch: int, seq: int, num_heads: int, head_dim: int):
    k = torch.randn(batch, seq, num_heads, head_dim)
    v = torch.randn(batch, seq, num_heads, head_dim)
    return torch.stack([k, v], dim=0)


def _make_zero_layer_kv(batch: int, seq: int, num_heads: int, head_dim: int):
    k = torch.zeros(batch, seq, num_heads, head_dim)
    v = torch.zeros(batch, seq, num_heads, head_dim)
    return torch.stack([k, v], dim=0)


def test_v1_engine_roundtrip_full_hit():
    from distributed_kv_manager.engine.v1 import init_v1_engine, destroy_v1_engine

    with tempfile.TemporaryDirectory() as temp_dir:
        vcfg = _VLLMConfig(temp_dir)

        # build v1 engine instance
        core = init_v1_engine(vcfg)

        # build original KV caches and register
        batch, seq, num_heads, head_dim = 1, 16, 2, 8
        num_layers = 2
        orig_kvs: Dict[str, Any] = {
            f"layer_{i}": _make_layer_kv(batch, seq, num_heads, head_dim)
            for i in range(num_layers)
        }
        core.register_kv_caches(orig_kvs)

        # new request and scheduling
        tokens = list(range(seq))
        req = _Request("req1", tokens)
        # scheduler: update after alloc so engine tracks unfinished request
        core.update_state_after_alloc(req, None, 0)

        # build scheduler_output and forward_context
        num_sched = {req.request_id: len(tokens)}
        fc = _ForwardContext(req.model, orig_kvs)
        sched = SimpleNamespace(
            finished_req_ids=set(),
            scheduled_new_reqs=[req],
            num_scheduled_tokens=num_sched,
            scheduled_cached_reqs=[],
            forward_context=fc,
        )
        core.build_connector_meta(sched)

        # simulate compute completion for prefill
        req.num_computed_tokens = len(tokens)
        # trigger save
        core.wait_for_save()

        # reconstruct engine and perform retrieval into zero buffers
        destroy_v1_engine()
        core = init_v1_engine(vcfg)

        zero_kvs: Dict[str, Any] = {
            f"layer_{i}": _make_zero_layer_kv(batch, seq, num_heads, head_dim)
            for i in range(num_layers)
        }
        core.register_kv_caches(zero_kvs)

        # second request: same tokens, expect full hit minus last token allocation
        req2 = _Request("req2", tokens)
        need, can_load = core.get_num_new_matched_tokens(req2, num_computed_tokens=0)
        assert can_load is True
        assert need == max(0, len(tokens) - 1)

        core.update_state_after_alloc(req2, None, need)
        num_sched2 = {req2.request_id: len(tokens)}
        fc2 = _ForwardContext(req2.model, zero_kvs)
        sched2 = SimpleNamespace(
            finished_req_ids=set(),
            scheduled_new_reqs=[req2],
            num_scheduled_tokens=num_sched2,
            scheduled_cached_reqs=[],
            forward_context=fc2,
        )
        core.build_connector_meta(sched2)
        core.start_load_kv(fc2)

        # verify caches have been filled to match original
        for i in range(num_layers):
            name = f"layer_{i}"
            assert torch.allclose(zero_kvs[name], orig_kvs[name], atol=1e-6)

        destroy_v1_engine()
