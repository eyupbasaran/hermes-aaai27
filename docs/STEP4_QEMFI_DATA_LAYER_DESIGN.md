Step 4 should be treated as **a leakage-safe dataset boundary**, not merely a NumPy loader. The current Step 3 simulator boundary is the right pattern to copy: public `PendingJob` contains only predicted quantities, while true runtime/finish/cost are private in `_InternalPendingJob`.   The QeMFi layer should follow the same split: **public candidate/source/feature objects for schedulers; hidden label/runtime/rank objects for replay and evaluation only.**

QeMFi is a good first dataset because it has five TD-DFT fidelities—**STO-3G, 3-21G, 6-31G, def2-SVP, def2-TZVP**—and includes computation times for time-benefit multifidelity benchmarking. ([arXiv][1]) The Zenodo record lists version **1.1.0**, individual `.npz` files for molecules such as acrolein, alanine, dmabn, nitrophenol, o-hbdi, sma, thymine, urea, and urocanic, with the dataset published as an open dataset under CC BY 4.0. ([Doi][2])

# Step 4 goal

Implement a QeMFi data layer that produces four separate things:

1. **Public dataset bundle**
   Safe for scheduler constructors. Contains candidates, public features, source catalog, split IDs, and public runtime-prediction model metadata. Contains no labels, no true runtimes, no target ranks, no top-K membership.

2. **Replay oracle**
   Implements the existing `ReplayOracle` boundary. Schedulers must never receive it. The repo already documents that `ReplayOracle` is simulator-only and exposes hidden labels/runtimes only through `query_hidden(...)`.

3. **Initial observations**
   Completed observations at time zero. These are allowed to contain revealed labels/runtimes because every scheduler receives the same initial completed data through `reset(...)`.

4. **Hidden evaluator**
   Computes top-K recovery, best target score, discovery curves, and final metrics after a run. It must never be passed to schedulers.

The hard rule:

> **No scheduler-visible object may expose labels except already completed observations; no scheduler-visible object may expose true runtimes except already completed observations; no scheduler-visible object may expose final target rank, top-K membership, or hidden target labels.**

# 1. Exact public APIs to implement

## 1.1 Module layout

Add these files:

```text
src/hermes/data/qemfi_loader.py
src/hermes/data/qemfi_schema.py
src/hermes/data/qemfi_features.py
src/hermes/data/qemfi_splits.py
src/hermes/data/qemfi_evaluator.py
```

Keep hidden/replay classes in the data layer, but do not export them through broad public imports except where the experiment runner needs them. The scheduler modules should import only public bundle types, not the hidden replay store.

The current repo already has the central domain objects we should reuse: `Candidate`, `Source`, `ProposedJob`, `CompletedObservation`, `ResourceBudget`, and `SchedulerState`.   Do not create a parallel set of candidate/source/job types inside the QeMFi loader.

---

## 1.2 `QeMFiDatasetConfig`

Purpose: validated config object parsed from `configs/dataset/qemfi*.yaml`.

Fields:

```text
name
version
data_root
cache_root
molecule_names
file_pattern
source_order
target_source_id
property_spec
feature_spec
runtime_spec
split_spec
leakage_policy
```

Important subfields:

```text
property_spec:
  key
  component
  utility_direction
  allow_nan_policy

feature_spec:
  mode
  max_atoms
  normalize
  cache_version

runtime_spec:
  mode
  runtime_key
  public_runtime_model
  candidate_specific_public_runtime
  fallback_source_means

split_spec:
  split_seed
  candidate_pool_size
  initial_seed_size
  initial_source_policy
  require_all_sources
  candidate_id_policy

leakage_policy:
  forbid_label_like_candidate_metadata
  forbid_runtime_like_candidate_metadata
  forbid_rank_like_candidate_metadata
  expose_source_availability
```

Default policy should be strict:

```text
candidate_specific_public_runtime = false
forbid_label_like_candidate_metadata = true
forbid_runtime_like_candidate_metadata = true
forbid_rank_like_candidate_metadata = true
expose_source_availability = false
```

The current config already has the right five source IDs and target source, but `property_name` is still `null`; Step 4 should replace this with a structured `property_spec`.

---

## 1.3 `QeMFiLoadResult`

The loader should return a single object with clearly separated public and private parts:

```text
QeMFiLoadResult
  public_data
  replay_oracle
  runtime_model
  initial_observations
  evaluator
  load_report
```

Meaning:

```text
public_data:
  safe for scheduler constructors

replay_oracle:
  simulator-only; hidden labels and hidden true runtimes

runtime_model:
  public predicted runtime model, e.g. source mean + backend multiplier

initial_observations:
  already revealed CompletedObservation list

evaluator:
  hidden post-run metrics object

load_report:
  safe debug metadata: counts, selected molecules, feature mode, source IDs
```

`public_data` and `runtime_model` may be passed to schedulers. `replay_oracle` and `evaluator` must not.

---

## 1.4 `QeMFiPublicData`

Safe scheduler-facing object.

Fields:

```text
dataset_name
dataset_version
target_source_id
sources
candidates
candidate_ids
feature_names
split
public_runtime_summary
metadata
```

Allowed methods:

```text
get_candidates(candidate_ids=None)
get_features(candidate_ids)
get_source(source_id)
get_source_ids()
num_candidates()
```

Forbidden fields and methods:

```text
labels
y
target_y
raw_y
runtimes
true_runtime
rank
target_rank
topk
top_k
oracle
replay_oracle
evaluator
query_hidden
```

`Candidate.metadata` should be aggressively sanitized. Allowed metadata:

```text
dataset_name
molecule_name
raw_file_name
raw_index
raw_id
conformation_id
num_atoms
formula_or_atom_counts
```

Forbidden metadata:

```text
label
labels
y
target
target_y
runtime
time
rank
top
topk
score
utility
high_fidelity_value
```

The public candidate features must be computed only from public structural information such as atomic numbers and coordinates, not from QeMFi property arrays.

---

## 1.5 `QeMFiSplit`

Public split object.

Fields:

```text
split_seed
pool_candidate_ids
initial_candidate_ids
initial_pairs
active_pool_candidate_ids
molecule_names
source_ids
target_source_id
```

Forbidden fields:

```text
topk_candidate_ids
top1_candidate_ids
target_ranks
target_labels
target_scores
hidden_labels
hidden_runtimes
```

The split may reveal which candidate/source pairs are initially observed because those observations will be revealed to all schedulers. It must not reveal which candidates are good.

---

## 1.6 `QeMFiReplayOracle`

Private simulator-facing object implementing the existing `ReplayOracle` protocol.

It should support only:

```text
query_hidden(candidate_id, source_id) -> OracleRecord
```

The returned `OracleRecord` already has the correct fields: hidden `y` and `true_runtime_seconds`.

Design constraints:

```text
No public .labels property
No public .runtimes property
No public .target_ranks property
No public .topk property
No bulk getter
No dataframe getter
No get_all_records()
```

Private internal arrays may exist with leading underscores:

```text
_labels_by_candidate_source
_runtimes_by_candidate_source
_raw_labels_by_candidate_source
_candidate_id_to_row
```

The underscore does not create real security, but it enforces project discipline and makes tests easier.

---

## 1.7 `QeMFiHiddenEvaluator`

Private post-run evaluator.

Allowed methods:

```text
compute_final_metrics(completed_observations)
compute_discovery_curve(completions)
compute_topk_recovery(completed_observations, k_fraction)
compute_best_target_score(completed_observations)
```

It may internally hold:

```text
target_labels
target_ranks
topk_sets
```

But it must never be reachable from:

```text
QeMFiPublicData
Candidate.metadata
SchedulerState
BaseScheduler.reset
BaseScheduler.update
BaseScheduler.propose
```

This follows the Step 3 scheduler API boundary: schedulers currently receive only `reset(seed, initial_observations)`, `update(completed_jobs)`, and `propose(state, available_slots)`.

---

# 2. QeMFi raw-data handling

## 2.1 Files and molecules

The loader should support one or more QeMFi `.npz` files. Zenodo lists files like:

```text
QeMFi_acrolein.npz
QeMFi_alanine.npz
QeMFi_dmabn.npz
QeMFi_nitrophenol.npz
QeMFi_o-hbdi.npz
QeMFi_sma.npz
QeMFi_thymine.npz
QeMFi_urea.npz
QeMFi_urocanic.npz
```

The Zenodo page lists the dataset files and sizes, with total listed files of about 395 MB. ([Doi][2])

Use QeMFi’s own loading convention:

```text
np.load(path, allow_pickle=True)
```

The upstream README uses `allow_pickle=True` and shows keys such as `ID`, `R`, `Z`, `CONF`, `SCF`, `EV`, `TrDP`, `fosc`, `DPe`, `DPn`, `RCo`, and `DPRo`. ([GitHub][3])

---

## 2.2 Source order

Use this fixed source order:

```text
0 -> sto3g    -> STO-3G
1 -> 321g     -> 3-21G
2 -> 631g     -> 6-31G
3 -> def2svp  -> def2-SVP
4 -> def2tzvp -> def2-TZVP
```

`def2tzvp` is the default target source.

This matches both the official QeMFi fidelity list and the repo config already scaffolded for HERMES. ([arXiv][1])

---

## 2.3 Property selection

The loader should not hard-code one QeMFi property. It should support:

```text
property.key
property.component
property.utility_direction
```

Examples:

```text
fosc, component [0], maximize
EV, component [0], configurable direction
SCF, component [], usually minimize after utility transform
DPe, component [0], configurable
DPn, component [0], configurable
```

The upstream README shows that a property like `fosc` can be indexed as `array[:, fidelity_index, excitation_index]`, e.g. first excitation at STO-3G or second excitation at SVP. ([GitHub][3]) Therefore the loader must support properties whose arrays are:

```text
(n_candidates, n_sources)
(n_candidates, n_sources, n_components)
(n_candidates, n_sources, ...)
```

The config’s `component` should select dimensions after the source axis.

---

## 2.4 Utility transform

The hidden oracle should return a **utility value**, not necessarily the raw QeMFi value.

Use:

```text
if utility_direction == maximize:
    utility_y = raw_y

if utility_direction == minimize:
    utility_y = -raw_y
```

Do not standardize using the full target distribution before replay. Full-distribution standardization can leak global target statistics. If standardization is needed later, fit it only on completed observations inside the surrogate.

The hidden evaluator may keep both raw values and utility values, but schedulers should only see completed utility observations.

---

## 2.5 Runtime handling

QeMFi includes computation times for time-benefit benchmarking. ([arXiv][1]) Step 4 must store true runtimes privately and expose only predicted runtime models publicly.

Runtime modes:

```text
require_real:
  Main experiment mode. Loader must find real runtime data or raise a clear error.

synthetic_debug:
  Unit-test/debug mode only. Creates deterministic source-level runtimes.

source_mean_public:
  Scheduler-visible runtime model uses source-level mean runtimes, not candidate-specific true runtimes.

learned_runtime_public:
  Later mode. Runtime predictor may use only runtimes observed through completed jobs.
```

For Step 4, default main config should be:

```text
runtime.mode = require_real
runtime.public_runtime_model = source_mean
runtime.candidate_specific_public_runtime = false
```

The public `RuntimeModel.predict(candidate_id, source_id, backend_id)` may accept `candidate_id`, but in the default public mode it must not use hidden per-candidate runtime. A test should verify that two candidates with the same source/backend receive the same predicted runtime in `source_mean_public` mode.

If the exact runtime key in QeMFi `.npz` is not obvious, implement a schema inspector that reports all keys and shapes, then either maps the configured runtime key or raises. Do not silently fall back to fake runtimes in main mode.

---

# 3. Feature handling

Step 4 should avoid RDKit and heavy QML dependencies. The upstream QeMFi repository provides scripts for Coulomb matrices and SLATM representations, but Step 4 does not need to depend on the upstream script stack. ([GitHub][3])

Default feature mode:

```text
geometry_basic_v1
```

Computed only from:

```text
Z: atomic numbers
R: coordinates
CONF: conformation ID or geometry index
```

Feature components:

```text
atom-count histogram
num_atoms
heavy_atom_count
coordinate centroid / spread
pairwise distance summary
Coulomb-matrix eigenvalues, padded/truncated to max_atoms
optional molecule-name one-hot disabled by default
```

Do **not** use these as features:

```text
SCF
EV
TrDP
fosc
DPe
DPn
RCo
DPRo
any low-fidelity label
any high-fidelity label
any runtime
any rank
any target top-K membership
```

Low-fidelity values must enter the model only after the scheduler actually queries that source and the simulator returns a completed observation.

Cache policy:

```text
cache_root/qemfi/{feature_mode}/{raw_file_hash}/features.npz
```

The feature cache may contain:

```text
candidate_id
features
feature_names
public metadata
```

The feature cache must not contain:

```text
labels
runtimes
ranks
top-K sets
oracle records
```

# 4. Split logic

## 4.1 Candidate universe

The loader should build an eligible universe by filtering candidates with:

```text
finite selected property value for required sources
finite true runtime for required sources
valid geometry/features
```

This eligibility filter is allowed because failed/missing rows cannot be replayed. But the public object should not expose detailed failure masks by source if those masks are label/runtime-derived.

## 4.2 Stable candidate IDs

Use deterministic IDs independent of labels:

```text
candidate_id = stable hash or sequential remap of:
  molecule_name
  raw_file_name
  raw row index
  raw ID if available
  conformation ID if available
```

The candidate ID must not depend on:

```text
property value
target rank
runtime
top-K membership
```

## 4.3 Pool sampling

Default:

```text
candidate_pool_size = 10000
split_seed = fixed integer
pool_strategy = uniform_without_replacement
```

The sampling must not use target labels or low-fidelity labels. A test should permute all labels while keeping IDs/features fixed and verify that `pool_candidate_ids` and `initial_pairs` are unchanged.

## 4.4 Initial seed observations

Recommended default for M0:

```text
initial_source_policy = all_sources_for_seed_candidates
initial_seed_size = 100
```

Meaning:

```text
sample 100 seed candidates uniformly from the pool;
reveal all five source observations for those candidates as initial completed observations.
```

This helps source-reliability estimation without making the scheduler cheat, because all schedulers receive the same completed observations at time zero.

If target-query budget is enabled, initial target observations should count toward the target-query budget. The current simulator already subtracts completed and pending target-source jobs when computing remaining target queries.

Later we can test alternatives:

```text
target_only
all_low_plus_target_subset
low_only_then_target
```

But Step 4 should implement only the clean default plus config validation.

## 4.5 Hidden top-K

Compute target ranks and top-K sets only inside `QeMFiHiddenEvaluator`.

Public split must not contain:

```text
top1_ids
top5_ids
target_rank
is_topk
```

Metrics can use:

```text
top_1_percent_recovery
top_5_percent_recovery
best_target_utility
area_under_discovery_curve
```

But those are post-run outputs, never scheduler inputs.

# 5. Oracle boundary

The loader should construct an internal mapping:

```text
(candidate_id, source_id) -> OracleRecord(
    y = utility_y,
    true_runtime_seconds = true_source_runtime,
    metadata = private minimal metadata
)
```

Only the simulator calls `query_hidden(...)` after a proposed job has passed validation and is launched. This preserves the Step 3 rule that budget validation happens before hidden oracle lookup.

Completed observations may reveal:

```text
candidate_id
source_id
backend_id
observed utility y
observed runtime_seconds
observed cost
start_time
finish_time
```

Pending jobs may reveal only:

```text
candidate_id
source_id
backend_id
predicted_runtime_seconds
predicted_cost
predicted_finish_time
```

The current simulator snapshot already deep-copies visible pending jobs and completed observations into `SchedulerState`, and it constructs public backend states from visible pending jobs only.  Step 4 should not weaken this.

# 6. Test plan

Add these files:

```text
tests/test_qemfi_schema.py
tests/test_qemfi_loader_synthetic.py
tests/test_qemfi_split_logic.py
tests/test_qemfi_feature_safety.py
tests/test_qemfi_oracle_boundary.py
tests/test_qemfi_runtime_leakage.py
tests/test_qemfi_evaluator_privacy.py
tests/test_qemfi_config_validation.py
tests/test_qemfi_simulator_integration.py
```

All tests should use tiny synthetic `.npz` fixtures generated inside temporary directories. Do not download the real Zenodo dataset in unit tests.

---

## 6.1 Schema tests

Test that the loader accepts a synthetic `.npz` with keys:

```text
ID
R
Z
CONF
fosc
runtime
```

and maps:

```text
source index 0 -> sto3g
source index 1 -> 321g
source index 2 -> 631g
source index 3 -> def2svp
source index 4 -> def2tzvp
```

Test property selection:

```text
fosc[:, source_index, 0]
```

Test that unsupported property keys raise a clear error.

Test that invalid source-axis length raises a clear error.

Test that non-finite labels/runtimes are either filtered or rejected according to `allow_nan_policy`.

---

## 6.2 Loader synthetic tests

Use a small synthetic dataset:

```text
n_candidates = 20
n_sources = 5
property shape = (20, 5, 2)
runtime shape = (20, 5)
```

Assert:

```text
load_result.public_data.num_candidates() == candidate_pool_size
len(load_result.initial_observations) == initial_seed_size * number_of_seed_sources
all initial observations are CompletedObservation
all source IDs match configured source catalog
target source is def2tzvp
```

---

## 6.3 Split tests

Required tests:

```text
same split_seed -> same pool_candidate_ids and initial_pairs
different split_seed -> different pool_candidate_ids with high probability
candidate_pool_size > eligible universe -> clear error unless allow_smaller_pool is true
initial_seed_size > candidate_pool_size -> clear error
```

Leakage-specific split test:

```text
If labels are randomly permuted but IDs/features are unchanged,
pool_candidate_ids and initial_pairs remain unchanged.
```

This catches accidental label-based sampling.

---

## 6.4 Feature-safety tests

Recursively inspect every public object reachable from `QeMFiPublicData`.

Forbid keys/fields containing:

```text
label
labels
y
target_y
runtime
true_runtime
time_seconds
rank
top
topk
score
utility
oracle
hidden
```

Exception: public runtime summary may contain source-level aggregate names if explicitly allowed, but never candidate-specific true runtimes.

Test that `Candidate.metadata` contains only safe fields.

Test that feature arrays do not equal or encode label arrays in the synthetic fixture. For the synthetic fixture, make labels a simple obvious vector and assert no feature column is identical to any source label vector.

---

## 6.5 Oracle-boundary tests

Test that:

```text
QeMFiPublicData has no replay_oracle attribute
QeMFiPublicData has no evaluator attribute
QeMFiPublicData has no labels/runtimes arrays
QeMFiReplayOracle is not returned inside public_data.metadata
```

Test `query_hidden(...)`:

```text
oracle.query_hidden(candidate_id, source_id) returns expected OracleRecord
unknown candidate/source pair raises KeyError
```

Test no bulk hidden access:

```text
oracle has no public get_all_labels()
oracle has no public labels property
oracle has no public target_ranks property
```

---

## 6.6 Runtime-leakage tests

Default runtime mode should be `source_mean_public`.

For two different candidates with the same source/backend:

```text
runtime_model.predict(candidate_a, source, backend)
==
runtime_model.predict(candidate_b, source, backend)
```

unless a future config explicitly enables learned candidate-specific runtime from completed observations.

Test that public `Candidate.metadata` has no runtime fields.

Test that hidden true runtimes are revealed only through:

```text
CompletedObservation.runtime_seconds
```

after simulator completion.

Test that a scheduler constructed with `public_data` and `runtime_model` cannot infer the true runtime order of candidate-specific jobs before launch.

---

## 6.7 Evaluator privacy tests

The hidden evaluator may compute:

```text
top-1% recovery
top-5% recovery
best target utility
target query count
discovery curve
```

But test that:

```text
evaluator is not reachable from public_data
evaluator is not passed to scheduler.reset
evaluator is not passed to scheduler.propose
public split has no top-K candidate IDs
public candidate metadata has no rank/top-K flag
```

Use a `LeakageProbeScheduler` like Step 3 to recursively inspect all scheduler-visible inputs.

---

## 6.8 Config validation tests

Test invalid configs:

```text
target_source_id not in sources -> error
two sources marked target -> error
no source marked target -> error
property.component out of range -> error
runtime.mode=require_real but no runtime key exists -> error
candidate_specific_public_runtime=true with leakage strict mode -> error
property_name null or property.key null -> error
```

The existing `qemfi.yaml` and `qemfi_debug.yaml` currently leave `property_name` null, so Step 4 should fail fast until the new `property_spec` is configured.

---

## 6.9 Simulator integration test

Build synthetic QeMFi load result, then run:

```text
EventSimulator(
  replay_oracle = load_result.replay_oracle,
  runtime_model = load_result.runtime_model,
  sources = load_result.public_data.sources,
  backends = test_backends
)
```

Use a scripted scheduler that receives:

```text
public_data
runtime_model
```

in its constructor, then proposes a small number of jobs.

Assert:

```text
scheduler never sees hidden labels
scheduler never sees true runtimes while pending
completed observations reveal labels only after completion
hidden evaluator computes metrics after run
```

# 7. Needed config changes

Replace the current dataset configs with structured QeMFi configs. Current source definitions are good; keep those IDs.

Recommended `configs/dataset/qemfi_debug.yaml` design:

```yaml
dataset:
  name: qemfi
  version: zenodo_1.1.0
  data_root: data/qemfi
  cache_root: data/cache/qemfi_debug

  raw:
    file_pattern: QeMFi_{molecule}.npz
    molecules:
      - urea

  npz_keys:
    id: ID
    coordinates: R
    atomic_numbers: Z
    conformation: CONF

  sources:
    - source_id: sto3g
      name: STO-3G
      qemfi_index: 0
      is_target: false
      nominal_cost_rank: 1
    - source_id: 321g
      name: 3-21G
      qemfi_index: 1
      is_target: false
      nominal_cost_rank: 2
    - source_id: 631g
      name: 6-31G
      qemfi_index: 2
      is_target: false
      nominal_cost_rank: 3
    - source_id: def2svp
      name: def2-SVP
      qemfi_index: 3
      is_target: false
      nominal_cost_rank: 4
    - source_id: def2tzvp
      name: def2-TZVP
      qemfi_index: 4
      is_target: true
      nominal_cost_rank: 5

  target_source: def2tzvp

  property:
    key: fosc
    component: [0]
    utility_direction: maximize
    allow_nan_policy: drop_candidate_if_any_required_source_missing

  runtime:
    mode: require_real
    runtime_key: null
    runtime_key_candidates:
      - runtime
      - runtimes
      - time
      - times
      - walltime
      - comp_time
      - computation_time
    public_runtime_model: source_mean
    candidate_specific_public_runtime: false
    synthetic_debug_allowed: false

  features:
    mode: geometry_basic_v1
    max_atoms: auto
    normalize: true
    cache_version: v1

  split:
    split_seed: 0
    candidate_pool_size: 1000
    initial_seed_size: 50
    initial_source_policy: all_sources_for_seed_candidates
    require_all_sources: true
    candidate_id_policy: stable_molecule_row_id

  leakage:
    forbid_label_like_candidate_metadata: true
    forbid_runtime_like_candidate_metadata: true
    forbid_rank_like_candidate_metadata: true
    expose_source_availability: false
```

Recommended main config differences:

```yaml
raw:
  molecules:
    - acrolein
    - alanine
    - dmabn
    - nitrophenol
    - o-hbdi
    - sma
    - thymine
    - urea
    - urocanic

split:
  candidate_pool_size: 10000
  initial_seed_size: 100
```

If runtime discovery fails on real QeMFi files, do **not** silently switch to synthetic runtimes. Instead, the loader should print the available keys/shapes and fail with an instruction to set `runtime.runtime_key` or provide a runtime sidecar.

# 8. Implementation guidance for Codex

## Phase 4A: tests first

Ask Codex to create the QeMFi tests before implementing behavior.

Priority order:

```text
1. tests/test_qemfi_config_validation.py
2. tests/test_qemfi_loader_synthetic.py
3. tests/test_qemfi_feature_safety.py
4. tests/test_qemfi_oracle_boundary.py
5. tests/test_qemfi_runtime_leakage.py
6. tests/test_qemfi_split_logic.py
7. tests/test_qemfi_evaluator_privacy.py
8. tests/test_qemfi_simulator_integration.py
```

Do not use real QeMFi downloads in CI tests. Generate synthetic `.npz` files in `tmp_path`.

## Phase 4B: schema/config parsing

Implement the config dataclasses and strict validation.

Do not proceed to real data loading until config errors are crisp.

The first config failure to fix is `property_name: null`; it should become `property.key`, `property.component`, and `property.utility_direction`.

## Phase 4C: synthetic loader

Implement against synthetic `.npz` first.

The synthetic fixture should deliberately include:

```text
label arrays with obvious values
runtime arrays with obvious candidate-specific values
target ranks with obvious top candidates
```

Then tests should prove none of that leaks through `public_data`.

## Phase 4D: feature builder

Implement `geometry_basic_v1`.

Do not depend on RDKit, `qml`, ORCA, or QeMFi upstream scripts in Step 4.

Use only:

```text
R
Z
CONF
ID
```

for public candidates/features.

## Phase 4E: hidden oracle and evaluator

Implement `QeMFiReplayOracle` and `QeMFiHiddenEvaluator`.

Keep both out of public data.

The experiment runner can hold:

```text
load_result.replay_oracle
load_result.evaluator
```

but scheduler constructors should receive only:

```text
load_result.public_data
load_result.runtime_model
```

## Phase 4F: real-file smoke test

After synthetic tests pass, add a non-CI smoke command:

```text
python -m hermes.data.qemfi_loader --config configs/dataset/qemfi_debug.yaml --inspect
```

It should print:

```text
selected files
available npz keys
key shapes
selected property shape
runtime key decision
eligible candidate count
candidate pool count
initial observation count
feature dimension
public leak-check result
```

This should not print labels, ranks, top-K IDs, or candidate-specific runtimes.

# 9. Anti-leakage checklist for Step 4 completion

Declare Step 4 complete only when these are true:

```text
QeMFiPublicData contains no labels, runtimes, ranks, top-K flags, oracle, or evaluator.
Candidate.metadata contains no forbidden label/runtime/rank fields.
Public features are computed only from public structural data.
RuntimeModel default is source-level or otherwise non-candidate-specific.
ReplayOracle is simulator-only.
HiddenEvaluator is post-run only.
Split is reproducible and label-independent.
Initial observations are the only labels visible at t=0.
Synthetic tests prove label permutation does not change splits.
Synthetic tests prove true runtimes do not appear in public data.
Simulator integration test proves scheduler sees no QeMFi hidden state.
```

# 10. Codex handoff prompt

```text
Design and implement Step 4: the QeMFi data layer.

Do not start with real downloads. Write tests first using tiny synthetic .npz fixtures.

Core requirement:
The QeMFi layer must preserve the Step 3 simulator contract. Schedulers may receive public candidate features, source metadata, public split IDs, and a public runtime prediction model. Schedulers must never receive ReplayOracle, hidden labels, true runtimes before completion, target ranks, final top-K membership, or the hidden evaluator.

Implement:
- src/hermes/data/qemfi_schema.py
- src/hermes/data/qemfi_loader.py
- src/hermes/data/qemfi_features.py
- src/hermes/data/qemfi_splits.py
- src/hermes/data/qemfi_evaluator.py

Public API:
- QeMFiDatasetConfig
- QeMFiPublicData
- QeMFiSplit
- QeMFiLoadResult
- load_qemfi_from_config(config) -> QeMFiLoadResult

Private/simulator-only:
- QeMFiReplayOracle implementing ReplayOracle
- QeMFiHiddenEvaluator
- private hidden label/runtime/rank store

Use existing core types:
- Candidate
- Source
- ProposedJob
- CompletedObservation
- OracleRecord
- ReplayOracle
- RuntimeModel

Feature rule:
Use only public structural arrays such as R, Z, CONF, and ID. Do not use any QeMFi property arrays, source labels, target labels, runtimes, ranks, or top-K information as features or candidate metadata.

Runtime rule:
True runtimes belong only to the replay oracle/internal store. The default public runtime model is source-level mean runtime, not candidate-specific true runtime. If runtime.mode=require_real and no runtime key can be found, raise a clear error. Do not silently use synthetic runtimes in main mode.

Split rule:
Candidate pool and initial seed selection must be reproducible and independent of labels/runtimes/ranks. Add a test that permuting labels does not change split IDs.

Evaluator rule:
Top-K membership and target ranks live only inside QeMFiHiddenEvaluator and are used only after the simulation run to compute metrics.

Tests:
Add tests for schema validation, synthetic loader behavior, feature safety, split reproducibility, label-independent splitting, oracle boundary, runtime anti-leakage, evaluator privacy, and simulator integration.

Acceptance:
pytest -q must pass, and the public QeMFi objects must pass the same recursive no-secrets check used in Step 3.
```

[1]: https://arxiv.org/abs/2406.14149 "QeMFi: A Multifidelity Dataset of Quantum Chemical Properties of Diverse Molecules"
[2]: https://doi.org/10.5281/zenodo.13925688 "QeMFi: A Multifidelity Dataset of Quantum Chemical Properties of Diverse Molecules"
[3]: https://github.com/vivinvinod/QeMFi "GitHub - vivinvinod/QeMFi: Scripts for various things related to \"QeMFi: A Multifidelity Dataset of Quantum Chemical Properties of Diverse Molecules\". includes MFML learning curves, ORCA scripts etc. · GitHub"
