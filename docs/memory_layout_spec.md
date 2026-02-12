# Memory Layout Specification for Streaming HFT Pipeline

## 1) Scope and Goals

This specification defines **layout and memory-contract rules** for the typetrace data path:

1. Multi-instrument feed ingest
2. Rolling high-frequency feature calculations
3. 1-second universe concatenation
4. Optimizer execution (Numba first)
5. Evaluator backend swap (Numba / DrJit / Torch) on CPU and GPU

Primary goals:

- Make layout behavior explicit and testable
- Separate math/type contract from runtime execution constraints
- Eliminate hidden copies in hot paths
- Support backend swaps without changing semantic results

Out of scope:

- Strategy logic correctness
- Exchange/network transport protocol choices

---

## 2) Contract Model Split

### 2.1 `TypeDesc` (math contract)

`TypeDesc` describes what the tensor/array **means**, independent of backend/runtime.

| Field | Type | Meaning | Required |
|---|---|---|---|
| `dims` | `tuple[str, ...]` | Ordered logical axes (e.g., `("time", "symbol", "feature")`) | Yes |
| `shape` | `tuple[int, ...]` | Logical lengths matching `dims` order | Yes |
| `dtype` | `str` | Semantic element type (`float32`, `float64`, `int32`, `bool`) | Yes |
| `domain` | `str` | Data domain (`price`, `returns`, `signal`, `weights`) | No |
| `unit` | `str` | Physical/statistical unit (`USD`, `bps`, `zscore`) | No |
| `missing_policy` | `str` | `forbid` / `allow_nan` / `mask_required` | Yes |
| `time_semantics` | `str` | `event_time` / `wall_time` / `bar_close_time` | Yes |

#### `TypeDesc` invariants

- `len(dims) == len(shape)`
- all `shape[i] >= 0`
- `dims` names unique
- any op that changes axis order must update `dims`
- `dtype` must be backend-representable for chosen runtime (checked against `ExecutionTraits`)

### 2.2 `ExecutionTraits` (runtime contract)

`ExecutionTraits` describes how data is physically stored/executed.

| Field | Type | Meaning | Required |
|---|---|---|---|
| `device` | `Literal["cpu", "cuda"]` | Residency target | Yes |
| `layout` | `Literal["C", "F", "strided"]` | Physical order intent | Yes |
| `strides_bytes` | `tuple[int, ...]` | Physical stride per axis in bytes | Yes |
| `contiguous_c` | `bool` | C-contiguous flag | Yes |
| `contiguous_f` | `bool` | F-contiguous flag | Yes |
| `alignment_bytes` | `int` | Base pointer alignment expectation | Yes |
| `memory_form` | `Literal["SoA", "AoS"]` | Struct organization | Yes |
| `pinned_host` | `bool` | Host page-locked memory | Yes |
| `readonly` | `bool` | Runtime mutability guard | Yes |
| `owner` | `str` | Allocator/owner id (`numpy`, `numba`, `drjit`, `torch`) | Yes |
| `interop_handle` | `dict | None` | DLPack/CUDA array interface metadata | No |

#### `ExecutionTraits` invariants

- If `layout == "C"`, then `contiguous_c == True`
- If `layout == "F"`, then `contiguous_f == True`
- If `device == "cuda"`, `pinned_host` only valid for staging buffers (not device array)
- `alignment_bytes` in `{16, 32, 64, 128}` for SIMD/coalesced paths
- `strides_bytes` length equals tensor rank

### 2.3 Compatibility rules (`TypeDesc` x `ExecutionTraits`)

| Check | Rule | Action on failure |
|---|---|---|
| Rank match | `len(TypeDesc.shape) == len(ExecutionTraits.strides_bytes)` | Hard error |
| Dtype support | `TypeDesc.dtype` supported on backend/device | Hard error |
| Missing policy | If `forbid`, runtime data must be finite/no NaN | Hard error |
| Alignment | If op requires alignment (SIMD/GPU kernel), validate `alignment_bytes` | Copy to compliant buffer or hard error (configurable) |
| Contiguous requirement | If backend kernel needs contiguous, require C/F flag | Explicit materialize + emit copy metric |
| Mutability | If op mutates and `readonly=True` | Copy-on-write with warning metric |

---

## 3) Layout Semantics

### 3.1 Canonical axis conventions for this pipeline

- Feed/feature tensors use `("time", "symbol", "feature")`
- Optimizer matrix form uses `("sample", "feature")`
- Weight vectors use `("feature",)`

### 3.2 Preferred memory form

- **SoA required** for hot numerical kernels (rolling, matmul, reductions)
- AoS allowed only at ingest boundary (decode/parsing layer)
- Transition `AoS -> SoA` must occur once per ingest micro-batch

### 3.3 Contiguity and order policy

- CPU Numba kernels: default **C-contiguous**
- Column-oriented linear algebra kernels may request F-contiguous explicitly
- DrJit/Torch handoff: allow non-owning view only if strides/order are accepted by target backend; otherwise explicit re-layout

### 3.4 Alignment policy

- CPU SIMD-targeted arrays: minimum 64-byte alignment
- GPU-bound staging buffers: 64-byte aligned + pinned host
- Device arrays: allocator-native alignment; record observed alignment in traits

### 3.5 Placement and transfer boundaries

| Boundary | Source | Target | Zero-copy target | Notes |
|---|---|---|---|---|
| Ingest decode -> feature buffer | parser/AoS | numpy/SoA | No | Normalize once |
| xarray/pandas -> numpy | xarray/pandas | numpy ndarray | Rare | Usually copies due to index/object dtypes |
| numpy -> numba (CPU) | ndarray | ndarray view | Yes | If contiguous + dtype-compatible |
| numpy(host pinned) -> CUDA backend | host | device | No (transfer), overlap possible | Async transfer w/ streams |
| torch <-> drjit via DLPack | device tensor | device tensor | Yes (conditional) | Lifetime/ownership hazards |

### 3.6 Copy-on-write hazards

- Pandas/xarray selection often creates views that trigger copy during numeric cast
- Slicing non-contiguous arrays into mutation kernels may cause implicit temp allocation
- Backend interop through DLPack can invalidate source if ownership consumed
- Rule: all materializations must increment `layout_copy_counter` metric

---

## 4) Operation-Level Layout Transition Rules

### 4.1 Transition matrix

| Operation | Input requirement | Output layout | Copy risk | Rule |
|---|---|---|---|---|
| `sel/slice` | any strided | usually strided view | Medium | Preserve view; require explicit `ascontiguousarray` before contiguous-only kernels |
| `transpose` | rank>=2 | strided (usually non-contig) | Low immediate / high downstream | Never assume contiguity post-transpose |
| `reshape` | compatible element count | view if contiguous-compatible else copy | Medium | Permit only if stride-compatible; else explicit materialize |
| `restack` | list of homogeneous arrays | contiguous target | High | Preallocate target and copy once |
| `concat` | same dtype + all dims except concat axis | contiguous target | High | Enforce same order/contiguity before concat to avoid N temp copies |
| `stack/unstack` | aligned shapes | stack often new contiguous | Medium | Prefer stack into preallocated buffer |
| `matmul` | numeric contiguous or backend-supported stride | backend dependent | Medium | Normalize to kernel-preferred order once |
| `reduction` | numeric | reduced rank, contiguous preferred | Low/Medium | For non-contig, use tuned kernel or explicit compact |
| `rolling window` | time-major preferred | windowed strided view or compact block | High | For GPU/numba kernels, compact to contiguous window blocks |
| `handoff: xarray/pandas->numpy` | typed numeric columns | ndarray | High | Reject object dtype, normalize index, force dtype at boundary |

### 4.2 Detailed guardrails by op

- **slice/sel**
  - Allowed: non-copy view when stride changes acceptable
  - Required before JIT kernel: contiguous check + explicit conversion if needed
- **transpose**
  - Treated as metadata operation only
  - Any subsequent mutating kernel must re-check writeable + contiguity
- **concat/re-stack**
  - Always allocate destination once (`np.empty`) and copy segments in order
  - Avoid chained concat in loops
- **rolling windows**
  - Use view-based rolling for analytics-only path
  - Use compact window tensor for optimizer features

---

## 5) Backend-Specific Constraints and Risks

### 5.1 Numba (CPU)

- Best path: `float32/float64`, C-contiguous ndarrays
- Unsupported/slow path: object dtypes, mixed-type arrays
- JIT warmup required; first-call latency must be excluded from steady-state benchmark
- Cache with `cache=True` where stable signatures exist
- Risk: silent object mode fallback (must fail build/tests if detected)

### 5.2 Numba (CUDA)

- Host->device transfer dominates at small batch sizes
- Prefer pinned host buffers and async copies
- Kernel launch overhead significant for tiny windows; batch by 1s universe chunk
- Dtype constraints tighter than NumPy generality (`float32` preferred)
- Risk: hidden synchronization points from accidental host access

### 5.3 DrJit (CPU/GPU)

- Vectorized symbolic kernels prefer static-ish shapes in hot loops
- Interop through DLPack/CUDA array interface may be zero-copy only with matching dtype/device/layout
- JIT graph construction cost amortized over repeated execution
- Risk: ownership/lifetime mistakes when exchanging buffers; define single owner after handoff

### 5.4 Torch evaluator (CPU/GPU)

- `torch.from_numpy` can share memory on CPU for compatible contiguous arrays
- CUDA tensors require explicit transfer unless originating from shared device handle
- Autograd should be disabled for evaluator path unless explicitly needed
- Risk: implicit dtype promotion (`float64` <-> `float32`) creating copies

---

## 6) Acceptance Criteria

## 6.1 Functional/layout correctness

- [ ] Every pipeline stage emits both `TypeDesc` and `ExecutionTraits`
- [ ] No stage mutates `TypeDesc` semantic fields incorrectly (dims/shape/dtype invariants hold)
- [ ] Layout-changing operations log pre/post traits
- [ ] All backend handoffs perform compatibility checks before execution

## 6.2 No-hidden-copy guarantees

- [ ] Copy counter metrics are emitted at each boundary/operation
- [ ] Unit tests assert expected copy count for representative pipelines
- [ ] Unexpected materialization causes test failure in strict mode
- [ ] Benchmarks include bytes-copied and transfer-time breakdown

## 6.3 Performance guardrails

- [ ] Warmup paths separated from steady-state metrics
- [ ] CPU and GPU profiles collected for same scenario
- [ ] Regression threshold defined (e.g., +10% bytes copied or +5% latency fails CI benchmark gate)

---

## 7) Test Plan

### 7.1 Unit tests

- `TypeDesc` invariant checks (rank/dims uniqueness/dtype)
- `ExecutionTraits` invariant checks (contiguity/layout/alignment)
- Compatibility matrix tests (pass/fail cases by dtype/device/layout)
- Operation transition tests:
  - slice -> non-contig -> explicit compact -> kernel-ready
  - transpose -> matmul path normalization
  - concat across symbols with preallocated destination

### 7.2 Integration tests

- End-to-end micro-pipeline: ingest -> rolling -> concat -> optimizer
- Backend swap parity: Numba CPU vs DrJit GPU vs Torch evaluator within numeric tolerance
- Transfer-boundary tests with pinned/unpinned host buffers

### 7.3 Benchmark tests

- Steady-state throughput (messages/s, bars/s)
- P50/P99 stage latency
- bytes copied per second
- host<->device transfer time and overlap efficiency

---

## 8) Worked Example (Explicit Shapes and Layout Transitions)

Scenario:

- 512 symbols
- 20 features per symbol
- 200ms ingest ticks, rolled into 1s universe bars
- 60-second rolling horizon

### Step A: Ingest (AoS) -> SoA feature cube

- Input decode buffer (`AoS`): shape `(512,)` records with struct `{symbol_id, bid, ask, last, vol, ...}`
- Normalize to SoA numeric buffer:
  - `TypeDesc`: dims `("time", "symbol", "feature")`, shape `(1, 512, 20)`, dtype `float32`
  - `ExecutionTraits`: device `cpu`, layout `C`, strides `(512*20*4, 20*4, 4)`, contiguous_c `True`, alignment `64`, memory_form `SoA`

### Step B: Rolling 60s window build

After 60 seconds collected at 1s cadence:

- Window tensor:
  - `TypeDesc`: dims `("time", "symbol", "feature")`, shape `(60, 512, 20)`
  - `ExecutionTraits`: C-contiguous compact block for optimizer path

### Step C: Universe concat -> optimizer matrix

- Reshape/restack from `(60, 512, 20)` to samples/features:
  - samples = `60 * 512 = 30720`
  - output shape `(30720, 20)`
- If source is C-contiguous with `feature` as last axis, reshape is view-compatible:
  - `TypeDesc`: dims `("sample", "feature")`, shape `(30720, 20)`
  - `ExecutionTraits`: layout `C`, contiguous_c `True`, no copy

### Step D: Numba CPU optimizer

- Requirement: C-contiguous `float32`
- Input already compliant -> zero-copy call
- Warmup first invocation excluded from latency SLO

### Step E: Swap evaluator to GPU (DrJit)

- Stage host pinned buffer from matrix `(30720, 20)`
- Async transfer to device tensor
- Post-transfer traits:
  - `device = "cuda"`, `owner = "drjit"`, contiguous flag backend-native
- If DLPack handoff to Torch evaluator in same device/context and dtype matches -> zero-copy possible

Copy accounting for this example (steady state target):

1. AoS decode -> SoA normalize: 1 required copy
2. Rolling compact block maintenance: 0 incremental copies if ring-buffered compactly
3. Reshape to `(sample, feature)`: 0 copies
4. CPU->GPU transfer for evaluator: 1 transfer (not hidden)

Hidden-copy budget target: **0** beyond the two explicit movements above.

---

## 9) Implementation Roadmap (Phased)

### Phase 0: Observability (no behavior changes)

- Add traits logging hooks and copy counters at all stage boundaries
- Add strict-mode flag to fail on untracked materializations

### Phase 1: Contract introduction

- Introduce `TypeDesc`/`ExecutionTraits` data structures
- Wrap existing stage outputs without changing existing compute kernels

### Phase 2: Operation policy enforcement

- Add explicit guards for contiguous/alignment/dtype before each backend call
- Replace implicit concat/reshape patterns with preallocated/materialize-once patterns

### Phase 3: Backend handoff hardening

- Implement unified handoff adapters: numpy<->numba, numpy<->drjit, drjit<->torch
- Enforce ownership/lifetime policy and interop metadata checks

### Phase 4: CI hard gates

- Add copy-budget assertions and performance regressions to CI benchmarks
- Require backend parity integration tests for release branches

---

## 10) Checklist for Adoption

- [ ] All pipeline nodes emit and validate `TypeDesc`
- [ ] All runtime arrays emit and validate `ExecutionTraits`
- [ ] Every layout-changing op follows transition matrix rules
- [ ] Backend adapters enforce compatibility table
- [ ] Copy/transfer metrics visible in logs + benchmark reports
- [ ] Strict mode enabled in CI for no-hidden-copy regressions
