import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable


class EntryState(Enum):
    RESERVED = 0
    FETCHING = 1
    READY = 2
    EVICTED = 3


@dataclass
class PrefetchEntry:
    key: str
    state: EntryState = EntryState.RESERVED
    last_update: float = field(default_factory=time.time)
    hits_after_prefetch: int = 0  # Telemetry: true hits following prefetch
    wasted_prefetch: int = 0      # Telemetry: prefetched but not hit then evicted


class PrefetchBuffer:
    """
    Read-only prefetch buffer index tracking in-flight and ready objects.
    Data itself lives in the SSD cache; this buffer only manages states.
    No persistent metadata or refcount changes.
    """

    def __init__(self, capacity: int = 4096):
        self._cap = capacity
        self._map: Dict[str, PrefetchEntry] = {}
        self._lock = threading.RLock()
        # Simple telemetry
        self.hits = 0
        self.misses = 0
        self.prefetch_submitted = 0
        self.evictions = 0

    def reserve(self, key: str) -> bool:
        with self._lock:
            if key in self._map:
                return False
            if len(self._map) >= self._cap:
                # FIFO eviction of one EVICTED/oldest READY
                evict_key = next(iter(self._map))
                self._map.pop(evict_key, None)
                self.evictions += 1
            self._map[key] = PrefetchEntry(key=key, state=EntryState.RESERVED)
            self.prefetch_submitted += 1
            return True

    def mark_fetching(self, key: str):
        with self._lock:
            if key in self._map:
                self._map[key].state = EntryState.FETCHING
                self._map[key].last_update = time.time()

    def mark_ready(self, key: str):
        with self._lock:
            if key in self._map:
                self._map[key].state = EntryState.READY
                self._map[key].last_update = time.time()

    def mark_evicted(self, key: str):
        with self._lock:
            if key in self._map:
                self._map[key].state = EntryState.EVICTED
                self._map[key].last_update = time.time()

    def is_ready(self, key: str) -> bool:
        with self._lock:
            e = self._map.get(key)
            return e is not None and e.state == EntryState.READY

    def on_access(self, key: str, hit: bool):
        with self._lock:
            e = self._map.get(key)
            if hit:
                self.hits += 1
                if e and e.state == EntryState.READY:
                    e.hits_after_prefetch += 1
            else:
                self.misses += 1


class BudgetEstimator:
    """
    budget_bytes = front_load_time_sec * effective_bw_bytes_per_s
    effective_bw is a sliding average derived from observed downloads.
    """

    def __init__(self, alpha: float = 0.3, min_bw: float = 10e6):
        self._alpha = alpha
        self._bw = min_bw  # bytes/sec

    def update(self, bytes_count: int, duration_sec: float):
        if duration_sec <= 0:
            return
        inst = bytes_count / max(duration_sec, 1e-6)
        self._bw = self._alpha * inst + (1 - self._alpha) * self._bw

    def bandwidth(self) -> float:
        return self._bw

    def budget(self, front_load_time_sec: float) -> float:
        return max(0.0, front_load_time_sec) * self._bw


class RateLimiter:
    """Byte-level token bucket rate limiter."""

    def __init__(self, bytes_per_sec: float):
        self._rate = max(bytes_per_sec, 1e6)
        self._tokens = self._rate
        self._last = time.time()
        self._lock = threading.Lock()

    def set_rate(self, bytes_per_sec: float):
        with self._lock:
            self._rate = max(bytes_per_sec, 1e6)

    def try_consume(self, nbytes: int) -> bool:
        now = time.time()
        with self._lock:
            # Refill
            delta = now - self._last
            self._last = now
            self._tokens = min(self._tokens + delta * self._rate, self._rate * 2)
            if self._tokens >= nbytes:
                self._tokens -= nbytes
                return True
            return False


class IOAggregator:
    """
    Collect read requests in a short window, dedupe, and dispatch with
    bandwidth fraction and queue depth limits.
    """

    def __init__(
        self,
        fetch_fn: Callable[[str], Optional[bytes]],
        on_ready: Callable[[str], None],
        rate_limiter: RateLimiter,
        window_ms: int = 30,
        max_batch_bytes: int = 64 * 1024 * 1024,
        max_qd: int = 4,
    ):
        self._fetch_fn = fetch_fn
        self._on_ready = on_ready
        self._rl = rate_limiter
        self._window_ms = window_ms
        self._max_batch = max_batch_bytes
        self._max_qd = max_qd
        self._queue: List[str] = []
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, keys: List[str]):
        with self._lock:
            self._queue.extend(keys)

    def stop(self):
        self._stop = True
        self._thread.join(timeout=1.0)

    def _worker(self):
        while not self._stop:
            time.sleep(self._window_ms / 1000.0)
            batch = self._drain_window()
            if not batch:
                continue
            self._dispatch(batch)

    def _drain_window(self) -> List[str]:
        with self._lock:
            if not self._queue:
                return []
            # Dedupe preserving order
            seen: Set[str] = set()
            out: List[str] = []
            while self._queue and len(out) < 1024:  # cap per window
                k = self._queue.pop(0)
                if k not in seen:
                    seen.add(k)
                    out.append(k)
            return out

    def _dispatch(self, keys: List[str]):
        # Split into sub-batches respecting bytes budget via rate limiter
        # We don't know exact file sizes; approximate with a per-object cap to gate QD.
        # Here we just gate on QD and rate limiter per object.
        sem = threading.Semaphore(self._max_qd)

        def _task(k: str):
            with sem:
                # Try a conservative token ask; 4MB optimistic chunk
                chunk = 4 * 1024 * 1024
                if not self._rl.try_consume(chunk):
                    # backoff a little
                    time.sleep(self._window_ms / 1000.0)
                data = self._fetch_fn(k)
                if data is not None:
                    self._on_ready(k)

        threads: List[threading.Thread] = []
        for k in keys:
            t = threading.Thread(target=_task, args=(k,), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=2.0)


class PlanBuilder:
    """
    Minimal planner that, given a current key and a budget, proposes
    a small list of likely-next keys. The caller is responsible for
    constructing candidate keys (e.g., next layers) and filtering
    existing/cached ones.
    """

    def build(self, candidates: List[str], budget_bytes: int, est_obj_size: int = 8 * 1024 * 1024) -> List[str]:
        if budget_bytes <= 0:
            return []
        out: List[str] = []
        spent = 0
        for c in candidates:
            if spent + est_obj_size > budget_bytes:
                break
            out.append(c)
            spent += est_obj_size
        return out
