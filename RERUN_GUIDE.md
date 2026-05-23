# MIREDO Section-V Rerun — Authoritative Reproduction Runbook

**This is the canonical, self-contained runbook** for reproducing the paper's
Section-V results. You need nothing but this repository — no paper, no progress
journal, no author assistance. It targets the frozen production configuration:
`CIM_ACC_DEFAULT_SETUP` hardware, the scout/sweep solver, and FlexFact ON. The
expected per-phase results are inlined in §6 so you can verify a reproduction
without any external reference.

**Audience:** someone with *only this code repository* — no paper, no prior
context, no author assistance — who must set up a fresh machine and run the
**entire Section-V rerun to completion unassisted**, avoiding the operational
pitfalls we already hit (see §8, the error ledger).

`<REPO>` = the root of your clone of this MIREDO code repository (the directory
containing `run.py`, `Evaluation/`, `Architecture/`, `model/`,
`environment.yml`, `README.md`). All commands run from `<REPO>` with the conda
env active unless stated. Everything needed to *produce the results* is in this
one repo. Read `<REPO>/README.md` first for the pipeline overview (ONNX →
ZigZag baseline cache → MIREDO loop repr → spatial-scheme search → per-candidate
Gurobi MIP → simulator validation → per-layer/model comparison).

---

## 0. This guide's commands are authoritative (read before anything else)

This runbook is **self-contained**: the exact per-phase invocations in §5 are
the single source of truth. You do not need — and this repository does not ship
— any external `experiments/parsed_metrics/` provenance journal or paper.
Run the §5 commands **exactly as written**: same script, models, objectives,
layers, architecture, and budget. Do **not** infer arguments from a driver's
defaults or substitute your own — the defaults are fail-safe, but the commands
pin every knob explicitly so the run reproduces with no other reference. Each
phase also writes its own `config` block into its output JSON; after every phase
verify that block with the §5 post-phase gate (architecture, budget, no missing
cells, empty/benign `anomalies[]`). Two of the pitfalls in §8 came from trusting
a driver's defaults or stale prose over the pinned commands here — when in
doubt, the §5 command wins.

---

## 1. One-time setup on a fresh machine

Do all of this before any experiment. Skipping any item is a hard blocker.

### 1.1 Python environment
```sh
cd <REPO>
conda env create -f environment.yml      # creates env "MIREDO", Python 3.10.19
conda activate MIREDO
```
`gurobipy==12.0.0` is installed by the env (pip section of `environment.yml`).

### 1.2 Gurobi license (HARD BLOCKER)
`gurobipy` is only the client library; the MIP solver will not run without a
valid license. Obtain one (academic licenses are free) and make it visible —
`export GRB_LICENSE_FILE=/path/to/gurobi.lic` or place it at `~/gurobi.lic`.
Validated by the §1.7 R3 gate.

### 1.3 CACTI (energy model — must be built)
Vendored at `<REPO>/utils/Cacti_wrapper/cacti` (C++). Build it (`make` in that
directory; see its `README.md`). On first run CACTI auto-generates per-hardware
energy configs under `utils/Cacti_wrapper/self_gen/` — expect a slow first layer
while it populates; this is normal, not a hang.

### 1.4 Baselines (vendored in-repo; one needs a C++ build)
Adapters live in `<REPO>/Evaluation/{Zigzag_imc,CIMLoop,CoSA}/` plus
`Evaluation/common/BaselineProvider.py`:
- **ZigZag-IMC**: uses the `zigzag` Python package (in the env) +
  `Evaluation/Zigzag_imc/zigzag_adapter.py`. The first MIREDO run per
  (objective, model, architecture) runs ZigZag once to build a baseline cache
  (`utils/ZigzagUtils.py: zigzag_cache_prefix`). No extra build.
- **CoSA**: fully in-repo (`Evaluation/CoSA/`). Solved as a Gurobi MIP — also
  needs the §1.2 license. No extra build.
- **CiMLoop**: Timeloop+Accelergy backend is a **git submodule** at
  `Evaluation/CIMLoop/timeloop-accelergy-infra/`. After cloning:
  ```sh
  git submodule update --init --recursive
  ```
  then build Timeloop + install the Accelergy plug-ins per that submodule's own
  build instructions (`import timeloopfe.v4` must succeed and the Timeloop
  binary must be on PATH — the adapter prepends its bin/lib at runtime, so it
  must be compiled there). CiMLoop is **mandatory** for a complete reproduction (see §2
  invariant 6 — a missing baseline is a failure, not a partial). Validated by
  the §1.7 R4 gate.

### 1.5 Workloads (already in-repo — nothing to download)
ONNX models in `<REPO>/model/`: `resnet18.onnx`, `vgg19bn.onnx`, `alexnet.onnx`,
`mobilenetV2.onnx`, `EfficientNet-B0.onnx` (CNN suite, 174 conv layers total),
and `bert_base.onnx`, `gpt2_medium_block.onnx`, `tinyllama_block.onnx`
(transformer suite). Layer parsing is automatic from the ONNX (README step 1).

### 1.6 Smoke test (confirm setup before the long rerun)
```sh
conda activate MIREDO
python Evaluation/RunBaselineComparison.py --models resnet18 \
  --objectives EDP --baselines ws zigzag \
  --architecture CIM_ACC_DEFAULT_SETUP --timeLimit 60 --mipFocus 1 \
  -o smoke_test
```
Expect `output/smoke_test*/baseline_comparison.json` with
`config.architecture_key == "CIM_ACC_DEFAULT_SETUP"` (every driver writes the
registry key under `config.architecture_key`; the camelCase main drivers *also*
embed the full hardware-spec dict under `config.architecture`, which is **not** a
string — gate on `architecture_key`, never on `architecture`). If that JSON
appears with the right `architecture_key`, the environment is good. Delete
`output/smoke_test*`.

### 1.7 Pre-flight HARD gates (all must pass; any failure stops the run)
A failure is a HALT with an error to the operator — never a silent skip or a
degraded substitute.

- **R3 — Gurobi license:**
  `python -c "import gurobipy; gurobipy.Model()"` succeeds with no license error.
- **R4 — CiMLoop runnable:** `python -c "import timeloopfe.v4"` succeeds AND the
  Timeloop binary resolves on PATH. If not, HALT — do not run §5.2 with CiMLoop
  missing (it would emit an incomplete baseline column).
- **R5 — Disk headroom:** `df -h <REPO>/output` — one full rerun is ≈ 2 GB.
  Confirm free space ≫ that; old `logs_rerun_*` dirs are audit-frozen (no
  deletion without authorization).
- **CACTI energy auto-recompute sanity:** `default_setup.py` enlarged the
  Input/Output buffers, so `default_spec()` re-runs CACTI for those two SRAM
  levels. Confirm the new per-bit energies are physically sane (monotone with
  capacity, same order of magnitude) before the long run:
  ```sh
  python - <<'PY'
  from Evaluation.common.EvalCommon import make_accelerator
  # make_accelerator returns a CIM_Acc; its HardwareSpec is .source_spec
  # (NOT .spec); memory levels are .source_spec.memory_hierarchy.
  o = make_accelerator("CIM_ACC_TEMPLATE").source_spec
  n = make_accelerator("CIM_ACC_DEFAULT_SETUP").source_spec
  oh = {m.name: m for m in o.memory_hierarchy}
  for m in n.memory_hierarchy:
      if m.name not in ("Dram","Input_buffer","Output_buffer","Global_buffer"):
          continue
      a = oh[m.name]
      print(f"{m.name}: size {a.size_bits//8}->{m.size_bits//8} B; "
            f"bw {a.r_bw_bits_per_cycle}->{m.r_bw_bits_per_cycle}; "
            f"r_pJ {a.r_cost_per_bit_pJ:.4f}->{m.r_cost_per_bit_pJ:.4f}; "
            f"w_pJ {a.w_cost_per_bit_pJ:.4f}->{m.w_cost_per_bit_pJ:.4f}")
  PY
  ```
  Expect: Dram bw 64→128; Input_buffer 2048→8192 B; Output_buffer 2048→16384 B;
  Global_buffer unchanged.

### 1.8 Cache isolation (run once, immediately before Phase A)
Cross-run isolation is a **separate concern from `BYPASS=1`** (§2.1). The MIP
cache key is `(hw_fp, loopdim, objective, time_limit, mip_focus, ablation_flags)`
and does **NOT** encode the parallel config — so any entry written under an
older solver regime (e.g. the pre-fix 4-thread contention) would be silently
re-served under the 8-thread solver and undo the §5.2 fix. Neutralize by
**renaming, not deleting**:
```sh
cd <REPO>/output
mv .mip_cache.pkl .mip_cache.pkl.bak_pre_rerun   # if present
mv .acc_cache.pkl .acc_cache.pkl.bak_pre_rerun   # name-keyed; serves a
                                                          # stale accelerator after
                                                          # the default_setup change
```
The new hardware fingerprint auto-misses; the backups are audit-only, never
restored mid-rerun. The CIMLoop/CoSA baseline caches are keyed by architecture
name, so `CIM_ACC_DEFAULT_SETUP` runs land in their own subdirs and are
auto-isolated. Do not read or merge any older `logs_rerun_*` output or cache.

---

## 2. Non-negotiable invariants (the pitfalls — read before running)

These are not bugs; ignoring them silently corrupts results.

### 2.1 `BYPASS=1` is ONLY for cost/timing phases — the single trigger
A phase is **cost/timing** iff **wall-clock time, scheme-prune count, or solve
overhead is itself a measured outcome reported in the paper**. A cache hit
returns in ~2 ms vs a cold solve of seconds-to-minutes → a single hit destroys
the measurement *and* (for the LB/FlexFact toggles, whose key omits the toggle)
can serve a stale result. Only this class gets `MIREDO_BYPASS_MIP_CACHE=1`.

A phase is **quality** iff it reports solved objective values
(`simLatency / simEnergy / simEDP`) or their derivatives (per-layer ratios,
baseline-vs-MIREDO %, ablation effect on quality). Under the same key, a cache
hit returns the *identical* deterministic result as a cold solve — so within-run
duplicate shapes are **safe to cache** and bypassing them buys nothing while
costing ~7.6 min per redundant solve.

| Phase | § | Class | `BYPASS` |
|---|---|---|---|
| A-EDP / A-LAT (CNN main) | 5.2.1 | quality | **no** |
| A-iso (CoSA subset) | 5.2.1 | quality | no |
| B (transformer main) | 5.2.2 | quality | no |
| C (objective tradeoff) | 5.5.2 | quality | **no** (reuses A's cache) |
| D (ablation) | 5.4.2 | quality | no |
| E (sensitivity, 6-axis) | 5.5.1 | timing/convergence | **=1** |
| F-1/2/3 (dyn-LB, FlexFact, cost-quality, wall-time) | 5.6 | timing | **=1** |
| §5.3.3 cert (chain + anchors) | 5.3.3 | cert | **=1** |
| §5.4.1 profile / Phase G | 5.4.1 / 5.3.1-2 | analysis-only | n/a (no solve) |

> **Operational rule:** does this phase report wall-clock or scheme-prune count
> as a paper outcome? Yes → `MIREDO_BYPASS_MIP_CACHE=1`. No → no env var. Do NOT
> generalize the timing rule to all phases — that over-application to A-EDP cost
> ~9 h (§8 E1).

### 2.2 Always pass AND verify `--architecture`
Use `CIM_ACC_DEFAULT_SETUP` (CNN) or `CIM_ACC_DEFAULT_SETUP_TRANSFORMER`
(transformer/attention) on *every* driver. **All 7 camelCase drivers now default
to `CIM_ACC_DEFAULT_SETUP` / `--timeLimit 60`** (fail-safe by default), and
the four §5.6 kebab drivers already did — so omitting the flag no longer
silently selects legacy 2 KB/64-bit hardware. Still pass it explicitly
(defense-in-depth), and **transformer phases MUST pass `…_TRANSFORMER`** because
the CNN setup is now the default. After each run, open the output JSON and
confirm `config.architecture_key` before trusting any number (the camelCase
drivers also embed the full hardware-spec dict under `config.architecture`; the
string key to assert is `architecture_key`). (The legacy `CIM_ACC_TEMPLATE` is
kept in the registry only as an audit baseline.)

### 2.3 Per-candidate MIP budget = 60 s, uniform
`--timeLimit 60 --mipFocus 1` (camelCase) / `--time-limit 60 --mip-focus 1`
(§5.6 kebab; `RunTopKBudget` has no `--time-limit` — its `--budgets` ARE the
limits) everywhere. Never lower it to save time; never mix budgets in one
comparison. A corner that needs >60 s is a flagged result, not a reason to
substitute a longer-budget number.

### 2.4 Never kill a long phase
Phases legitimately run many hours (Phase A ≈ 22–24 h per objective). The only
timing limit is the per-candidate MIP budget; phase wall-clock has no cap.

### 2.5 Solver settings are frozen — do not change mid-rerun
The committed solve path (in `utils/Tools.py: auto_parallel_config` and
`utils/SolverTSS.py`) is:
- **scout = 8 threads/scheme × `cores//8` workers** (on a 16-physical-core box:
  8t × 2w). 8 threads is the proven solve depth — a single-variable bit-exact
  proof showed the pre-fix 4-thread + 4-way contention under-converged hard EDP
  MIPs; 16 threads gives no further gain (Gurobi B&C plateau).
- **sweep = 1 thread × `cores` workers** — the cheap wide cull for the ~98% of
  schemes that are presolve-infeasible junk.
- **`scout_size = min(num_schemes, 20)`** — a 202-instance curated-CNN audit
  found the true-winner max util-product rank = 15, so the winner stays in the
  8-thread scout arm with margin. Do NOT reintroduce an adaptive `scout_size` or
  a "many-schemes → 2 threads" branch.
- **`SolverTSS` feasibility prescreen `TimeLimit = 20` s** (raised from 7;
  Conv_10/15 needed ~10.5 s to find their first feasible point and 7 s wrongly
  culled the genuinely-better scheme → §5.2 EDP suppression). Proven-infeasible
  schemes still die in presolve <1 s, so the cost is bounded.
- **`NoRelHeurTime = 10`** kept.

These are already set in code — leave them. (Optional opt-in instrumentation:
`MIREDO_FEAS_LOG=<path>` writes one line/scheme `{Gurobi status, feasibility_sec,
SolCount, scheme_dir}`; native behaviour is unaffected when unset.)

> **Benign console line — do not mistake for cache contamination:** Gurobi
> prints `Loaded MIP start from previous solve with objective …` on the second
> optimize of every scheme. This is NOT a cross-run/disk-persisted solution:
> `SolverTSS` runs the 20 s feasibility prescreen and then re-optimizes the
> *same in-process* `gurobipy.Model`, so Gurobi simply carries *this run's own*
> prescreen incumbent as the MIP start. No `.sol`/`.mst` is read from disk. Cold
> solves under `MIREDO_BYPASS_MIP_CACHE=1` remain genuinely cold.

### 2.6 No silent degradation — the rerun must be COMPLETE
All baselines × all layers; a failed cell is a failed reproduction, never an
acceptable partial. How a failure surfaces depends on the driver — so the
post-phase check differs by driver family:
- **Comparison / accuracy / ablation / sensitivity / tradeoff drivers**
  (Phase A/D and the EDP/latency/energy phases): each baseline or MIREDO failure
  is caught and recorded into the output JSON's `anomalies[]` array (`kind` =
  `baseline_error` / `runtime_error`, with the message) and the run continues.
  **Inspecting `anomalies[]` after every such phase is mandatory** — the driver
  will not stop for you.
- **§5.6 drivers** — failure surfaces three different ways here:
  - `RunAccelerationControl` / `RunFlexFactControl`: the solve is **not**
    wrapped, so any runtime failure aborts the driver with a traceback (non-zero
    exit) and there is no `anomalies[]`. Confirm exit 0 and a complete JSON with
    every expected row.
  - `RunTopKBudget`: the solve is likewise **unwrapped** (hard-fails on error),
    and the driver also appends a top-level `anomalies[]` of sanity checks —
    confirm exit 0, all cells present, and `anomalies[]` empty (it prints "No
    anomalies detected").
  - `RunBaselineWallTime`: each baseline **and** MIREDO call **is** wrapped — a
    failure is recorded in that result row's `error` field and the run
    continues. Confirm no row carries a non-benign `error`.
Across these solving-phase drivers the **only** benign failure is CoSA
auto-skipping depthwise/grouped layers — an inherent baseline-coverage limit
(recorded as `baseline_error`, identical every run, not a regression). Any other
`anomalies[]` entry, traceback, or row `error` is a HALT condition for *you*, the
operator: fix the root cause and rerun that phase; do **not** curate around a
missing/partial column. (The §5.4.1 traffic profile has its own documented benign
anomalies — see §6.)

### 2.7 FlexFact is ON
`MIREDO_DISABLE_FLEXFACT` must be **unset** (default = FlexFact ON, flexible
factor decomposition). Setting it to `1` forces prime-factor decomposition —
that is an ablation, not the production path. If unsure, verify the var is unset
via `/proc/<pid>/environ` on a running phase.

---

## 3. Hardware configuration (no paper needed to know this)

`--architecture CIM_ACC_DEFAULT_SETUP` resolves via
`Evaluation/common/EvalCommon.py` (`_ARCHITECTURE_SPEC_BUILDERS`) to
`Architecture/templates/default_setup.py`:

| Level | Parameter | Value |
|---|---|---|
| Chip | Cores | 8 |
| Chip | Global buffer | 256 KB, 128 bit/cycle |
| Chip | DRAM | 1 GB, 128 bit/cycle |
| Core | Input buffer | 8 KB, 128 bit/cycle |
| Core | Output buffer | 16 KB, 128 bit/cycle |
| Macro | Wordline × Bitline × Depth | 32 × 16 × 8 |
| Macro | Cells per weight | 8 |
| — | Tech / precision | 28 nm, INT8/INT8/INT16 |

OBuf = 2× IBuf reflects the 8 b activation in / 16 b psum out width asymmetry.
Transformer variant `CIM_ACC_DEFAULT_SETUP_TRANSFORMER` →
`Architecture/templates/default_setup_transformer.py` (16 cores, 64×32 macro,
4 MB GBuf, IBuf 16 KB / OBuf 32 KB, DRAM 256 bit/cycle). Do not edit these
templates; do not substitute the legacy `CIM_ACC_TEMPLATE`.

---

## 4. Case layers (used by several phases)

Defined in `Evaluation/common/CaseLayerShapes.py`. The four archetypes:
`L1` = ResNet-18 `Conv_8` (standard 3×3, C128 K128 G1);
`L2` = ResNet-18 `Conv_17` (deep 1×1, C256 K512 G1);
`L3` = MobileNet-v2 depthwise (a **synthetic** G=144 shape — no real model layer
has G=144; the real nearest is `Conv_19` G=192);
`L4` = EfficientNet-B0 MBConv 1×1 expansion (C=80, K=480, G1).

**Selector semantics (a known foot-gun):** bare `L1`..`L4` are *per-model ordinal
aliases* (the model's n-th layer), **not** the case registry. The case registry
is reached only via `--layerSource representative` (RunSensitivity) /
`--layer-ids` (the §5.6 kebab drivers) / `_annotate_representative_layers`. So a
plain `--layers L1..L4` does NOT reproduce a case-layer phase; check each phase's
exact selector below.

---

## 5. Run order + per-phase commands (must follow — phases have dependencies)

There is **no master script**; run the phases in this order. **Phase A must
finish first** because Phase G, the §5.4.1 profile, and §5.5.2 (Phase C, cache
reuse) depend on it. B/D/E/F and §5.3.3 are independent and may run in any order
once A is done.

```
A (§5.2.1)  RunBaselineComparison  CNN, 174 layers, EDP then Latency   ← run FIRST
A-iso       RunBaselineComparison  CoSA subset (3 nets, EDP)
B (§5.2.2)  RunBaselineComparison  transformer (3 models)
C (§5.5.2)  RunTradeoff            ResNet-only, Lat/Energy/EDP          (reuses A)
D (§5.4.2)  RunAblation            4 case layers, 2 structural variants
E (§5.5.1)  RunSensitivity         6-axis, L1–L4                        [BYPASS=1]
F (§5.6)    RunAccelerationControl / RunFlexFactControl /
            RunTopKBudget / RunBaselineWallTime  L1–L4                  [BYPASS=1]
G (§5.3.1/2) RunPhaseGAnalysis      analysis-only (REQUIRES A; pin root)
§5.4.1      extract_profiling_caselayer.py  (analysis; REQUIRES A + L1 pkl)
§5.3.3/§5.2.2 cert  VerifyOptimalityChain + VerifyBruteforce ×2         [BYPASS=1]
```

**The `-o` subrun contract (hardcoded):** `-o` is a path
component (`make_output_dir` → `output/<-o value>`, no auto subrun segment). The
analysis consumers hardcode EXACT subpaths, so each camelCase phase MUST use the
exact `-o` below. A wrong/missing subrun silently breaks Phase G and §5.4.1
(they `sys.exit` on a missing dir).

### Rerun output root (set once, before any phase)
Pick a name for this run's output directory. It MUST match the glob
`logs_rerun_*` so the analysis consumers (Phase G, §5.4.1) discover it. Export it
once; every command below writes under `output/$RUN/…`:
```sh
export RUN=logs_rerun_$(date +%Y%m%d)   # any logs_rerun_* name (e.g. logs_rerun_20260101)
export MIREDO_RERUN_ROOT="$RUN"         # pins the analysis consumers to this run
```

### Per-phase commands (run from `<REPO>`, env active)

> Flag conventions differ by driver. §5.2/§5.4.2/§5.5/§5.5.2 (`RunBaselineComparison`,
> `RunTradeoff`, `RunAblation`, `RunSensitivity`) use camelCase `--timeLimit
> --mipFocus` and `-o <section dir>`. The four §5.6 drivers
> (`RunAccelerationControl`, `RunFlexFactControl`, `RunTopKBudget`,
> `RunBaselineWallTime`) use kebab-case `--mip-focus`, `--layer-ids L1 L2 L3 L4`,
> and **`--output-json <explicit path>` — NOT `-o`**. Run
> `python Evaluation/<driver>.py --help` to confirm spelling before a long run.

**A §5.2.1 — CNN main (EDP + Latency).** Quality → **no BYPASS**. A single run
writes both objectives into the exact layout the downstream consumers hardcode —
`baseline_comparison/{EDP,Latency}/<net>/<layer>/…` plus
`baseline_comparison/baseline_comparison.json` (the driver builds
`<-o dir>/<objective>/<model>/<layer>/` per layer and the JSON at the `-o` root):
```sh
python Evaluation/RunBaselineComparison.py \
  --models resnet18 vgg19bn alexnet mobilenetV2 EfficientNet-B0 \
  --objectives EDP Latency --baselines zigzag ws cimloop cosa cosa_legal \
  --architecture CIM_ACC_DEFAULT_SETUP --timeLimit 60 --mipFocus 1 \
  -o $RUN/s5_2_1_cnn_main/baseline_comparison
```
Runs ~2 days (174 layers × 2 objectives; within-objective shape-twins cache-hit).
*(Optional: for a clean intermediate global-EDP checkpoint you may instead run two
passes — `--objectives EDP -o …/baseline_comparison_edp` then `--objectives Latency
-o …/baseline_comparison_lat` — and merge them into the same
`baseline_comparison/{EDP,Latency}/` + `baseline_comparison.json` layout. The
single run above already produces that layout, so the split is only an
operational convenience, not required.)*

**A-iso §5.2.1 — CoSA subset** (a distinct paper artifact, `cnn_cosa_iso.json`).
Quality → no BYPASS:
```sh
python Evaluation/RunBaselineComparison.py \
  --models resnet18 vgg19bn alexnet \
  --objectives EDP --baselines cosa cosa_legal \
  --architecture CIM_ACC_DEFAULT_SETUP --timeLimit 60 --mipFocus 1 \
  -o $RUN/s5_2_1_cnn_main/cosa_only
```

**B §5.2.2 — transformer main.** Quality → no BYPASS. Note `…_TRANSFORMER`:
```sh
python Evaluation/RunBaselineComparison.py \
  --models bert_base gpt2_medium_block tinyllama_block \
  --objectives EDP --baselines ws zigzag cimloop \
  --architecture CIM_ACC_DEFAULT_SETUP_TRANSFORMER --timeLimit 60 --mipFocus 1 \
  -o $RUN/s5_2_2_transformer/main
```

**C §5.5.2 — objective tradeoff (ResNet-only).** Quality, reuses A's cache →
**must NOT bypass**; A and C must share identical arch + 60 s. ResNet-only is a
**sanctioned design decision**, not a data gap — mobilenetV2 is dropped;
RunTradeoff loops per model so a single `--models resnet18` suffices:
```sh
python Evaluation/RunTradeoff.py --models resnet18 \
  --objectives Latency Energy EDP --baselines ws zigzag \
  --architecture CIM_ACC_DEFAULT_SETUP --timeLimit 60 --mipFocus 1 \
  -o $RUN/s5_5_2_tradeoff/network
```
RunTradeoff self-emits `objective_tradeoff.json` and no downstream script
hardcodes the tradeoff subpath, so the `/network` name is arbitrary. MIREDO
Latency/EDP cache-hit Phase A; Energy is the only genuinely new solve.

**D §5.4.2 — ablation, 4 case layers.** Quality → no BYPASS. **Pass the 4
explicit case-layer shapes via `--layers`** (exactly the four in the command
below — these are authoritative; an earlier guide said "full models", which was
WRONG and cost a 306-solve mis-launch, §8 E2). Variant
names are **HYPHENATED** (argparse `choices`-restricted; underscores hard-fail):
```sh
python Evaluation/RunAblation.py \
  --models resnet18 mobilenetV2 EfficientNet-B0 \
  --objectives Latency --structural fixed-double-buffer simplified-pipeline \
  --layers resnet18:Conv_8_3_3_28_28_128_128_1 \
           resnet18:Conv_17_1_1_7_7_256_512_1 \
           mobilenetV2:Conv_19_3_3_14_14_1_1_192 \
           EfficientNet-B0:Conv_30_1_1_14_14_80_480_1 \
  --architecture CIM_ACC_DEFAULT_SETUP --timeLimit 60 --mipFocus 1 \
  -o $RUN/s5_4_caselayer/ablation
```
Produces 9 rows = 3 models × {`latency-only` (the reference, 0% degradation),
`fixed-double-buffer`, `simplified-pipeline`}. The Latency pass cache-hits A; the
two structural variants set distinct `ablation_flags` (different key) → genuine
new solves. (`psum-capacity-only` is the paper-unused EXP-3c pilot — exclude.)
This `/ablation` subrun is also the hardcoded source for the §5.4.1 profile's
**L1** dataflow.

**E §5.5.1 — sensitivity, 6 axes.** **BYPASS=1.** Use `--layerSource
representative --layers L1 L2 L3 L4` (the case registry — NOT `--models`; the
`config.models:[resnet18]` recorded in `hardware_sensitivity.json` is an unused
default, a RED HERRING). Pass the explicit 6 `--parameters` (the driver's
`DEFAULT_SWEEPS` has a 7th `compartment_depth` not in the paper figure):
```sh
MIREDO_BYPASS_MIP_CACHE=1 python Evaluation/RunSensitivity.py \
  --layerSource representative --layers L1 L2 L3 L4 --baselines ws zigzag \
  --parameters core_count buffer_capacity gbuf_core_bw macro_spec \
               dram_bw operand_scratchpad \
  --architecture CIM_ACC_DEFAULT_SETUP --timeLimit 60 --mipFocus 1 \
  -o $RUN/s5_5_1_sensitivity/run
```

**F §5.6 — per-layer cost (4 kebab drivers).** **BYPASS=1.** Run sequentially
(no contention; §5.6.3 measures wall-time). `--output-json`, not `-o`:
```sh
# 5.6.1a dynamic load-balance control
MIREDO_BYPASS_MIP_CACHE=1 python Evaluation/RunAccelerationControl.py \
  --architecture CIM_ACC_DEFAULT_SETUP --time-limit 60 --mip-focus 1 \
  --layer-ids L1 L2 L3 L4 \
  --output-json output/$RUN/s5_6_perlayer_cost/5_6_1_dynlb_control.json
# 5.6.1b FlexFact ablation
MIREDO_BYPASS_MIP_CACHE=1 python Evaluation/RunFlexFactControl.py \
  --architecture CIM_ACC_DEFAULT_SETUP --time-limit 60 --mip-focus 1 \
  --layer-ids L1 L2 L3 L4 \
  --output-json output/$RUN/s5_6_perlayer_cost/5_6_1_flexfact_ablation.json
# 5.6.2 cost-quality (budgets ARE the per-cell limits; NO --time-limit)
MIREDO_BYPASS_MIP_CACHE=1 python Evaluation/RunTopKBudget.py \
  --architecture CIM_ACC_DEFAULT_SETUP --mip-focus 1 --layer-ids L1 L2 L3 L4 \
  --budgets 60 30 15 5 --top-ks all 10 5 3 \
  --output-json output/$RUN/s5_6_perlayer_cost/5_6_2_cost_quality.json
# 5.6.3 wall-time vs baselines — OMIT --methods (see §8 E3); code default is
#   [ws zigzag cimloop cosa cosa_legal miredo]; relabel cosa_legal→cosa-constrained at curation
MIREDO_BYPASS_MIP_CACHE=1 python Evaluation/RunBaselineWallTime.py \
  --architecture CIM_ACC_DEFAULT_SETUP --time-limit 60 --mip-focus 1 \
  --layer-ids L1 L2 L3 L4 \
  --output-json output/$RUN/s5_6_perlayer_cost/5_6_3_walltime.json
```

**G §5.3.1/2 — analysis-only.** No new solve. **Pin the rerun root** — "no-args"
resolves to the NEWEST `logs_rerun_*`, which would silently analyze the wrong
run (§8 E4):
```sh
MIREDO_RERUN_ROOT=$RUN python Evaluation/RunPhaseGAnalysis.py
```
Writes `_analysis/{5_3_1_fidelity,5_3_2_ranking}.json` + `diff_report.md`.

**§5.4.1 — traffic profile.** Analysis over A/D frozen dataflows.
**PRECONDITION:** L1 (resnet18 `Conv_8`) needs its Latency `Dataflow.pkl` at
`output/$RUN/s5_4_caselayer/ablation/Latency/resnet18/Conv_8_3_3_28_28_128_128_1/`. If
L1 was a CACHE HIT in A-LAT and D, **no pkl is written** — check first; if absent,
materialize it with one BYPASS solve, then run the profiler (pin the root):
```sh
# only if the L1 Dataflow.pkl is missing:
MIREDO_BYPASS_MIP_CACHE=1 python Evaluation/RunBaselineComparison.py \
  --models resnet18 --objectives Latency --baselines \
  --architecture CIM_ACC_DEFAULT_SETUP --timeLimit 60 --mipFocus 1 \
  --layers resnet18:Conv_8_3_3_28_28_128_128_1 \
  -o $RUN/s5_4_caselayer/ablation
MIREDO_RERUN_ROOT=$RUN python Evaluation/extract_profiling_caselayer.py
```
Known-benign anomalies: L3 ws/zigzag baseline-mapper mismatch (~10–11%, identical
every run, pre-existing) + `fresh_solve` notes.

**§5.3.3 / §5.2.2 cert — THREE distinct sub-steps (do NOT conflate).** All
**BYPASS=1**. They are easy to miss — run all three; the attention tile in
particular is NOT produced by `VerifyOptimalityChain` (§8 E5):
```sh
# (a) optimality chain + 7 conv anchors (CNN-only; TARGETS are hardcoded conv).
#     Do NOT use --architecture ..._TRANSFORMER here — it would solve CNN anchors
#     on transformer HW (meaningless).
MIREDO_BYPASS_MIP_CACHE=1 python Evaluation/VerifyOptimalityChain.py \
  --architecture CIM_ACC_DEFAULT_SETUP \
  --output-dir output/$RUN/_analysis
# (b) §5.3.3 1×1 C=K=64 anchor ("1409 cyc / 2.3e5 enumeration"; NOT in the chain TARGETS)
MIREDO_BYPASS_MIP_CACHE=1 python Evaluation/VerifyBruteforce.py \
  --case 1x1_C64K64 --arch CIM_ACC_DEFAULT_SETUP --objective both --timelimit 120
# (c) §5.2.2 QK^T attention tile (transformer HW)
MIREDO_BYPASS_MIP_CACHE=1 python Evaluation/VerifyBruteforce.py \
  --case attention_tiny --arch CIM_ACC_DEFAULT_SETUP_TRANSFORMER --objective both --timelimit 120
```
`VerifyBruteforce --arch` default is the dangerous legacy `CIM_ACC_TEMPLATE` —
always pass it. `VerifyBruteforce` has **no `--output-dir`**: it writes its
structured results to a fixed `output/Eval_Result/` and its log to a fixed
`output/brute_force_result.log` — neither is `$RUN`-scoped nor honours
`MIREDO_RERUN_ROOT`, so both the 1×1 (b) and attention (c) cert results land
there and **overwrite on a rerun**; copy them into `output/$RUN/_analysis/`
yourself if you need per-run isolation. Only chain step (a) honours
`--output-dir`, so its output goes to `output/$RUN/_analysis/`.

### Mandatory post-phase gate
After **every solving phase** (Phase A/D and the comparison/accuracy/ablation/
sensitivity/tradeoff/control/wall-time drivers), before any number is trusted or
curated: open each produced JSON and assert (1) `config.architecture_key` ==
intended (NOT `config.architecture`, which on the camelCase drivers is the full
spec dict — see §2.2), (2) the per-candidate budget == 60 s, and (3) no failed
cells — for the comparison/accuracy/ablation/sensitivity/tradeoff drivers that
means `anomalies[]` is empty or only the benign CoSA skips (§2.6); for the §5.6
drivers it means the driver exited 0 and emitted every expected row/cell, plus
(per §2.6) `RunTopKBudget`'s top-level `anomalies[]` is empty and no
`RunBaselineWallTime` row carries a non-benign `error`.
For the BYPASS phases also assert timing/scheme-count fields are populated (not
null). A mismatch invalidates the phase — rerun it, do not curate around it.

**Phase G (analysis) and the §5.3.3 / §5.2.2 cert sub-steps are gated
differently** (do not apply the rules above to them):
- *Phase G* (`RunPhaseGAnalysis` → `5_3_1_fidelity.json` / `5_3_2_ranking.json`)
  runs **no new solves** — it derives everything from the Phase A EDP output, so
  its JSONs carry a `provenance.source` pointer instead of
  `config.architecture_key`. Verify it by confirming `provenance.source` points
  at this run's Phase A output (already validated by the gate above), not by an
  arch field.
- *Cert sub-steps:* architecture is pinned by the `--arch` / `--architecture`
  flag on the cert command itself — both scripts HALT if the value is not in the
  registry, so a successful run already proves the right HW. Their result JSONs
  do **not** carry `config.architecture_key`; verify the arch from the command
  you ran (and the `on <arch>` line each script logs), not from a JSON field.
  Budget is **120 s by design** (steps (a)–(c) pass `--timelimit 120`) — the
  brute-force/enumeration reference budget, not the 60 s production budget. Do
  **not** flag the cert outputs against the 60 s rule. Everything else stays at
  60 s.

---

## 6. Outputs & completeness check

Canonical section dirs under `<REPO>/output/$RUN/`:
`s5_2_1_cnn_main/`, `s5_2_2_transformer/`, `s5_5_2_tradeoff/`,
`s5_4_caselayer/`, `s5_5_1_sensitivity/`, `s5_6_perlayer_cost/`, plus
`_analysis/` (Phase G + §5.3.3). A **solving-phase** JSON counts as produced if
it exists with the right `config.architecture_key`, a non-empty `results` block,
no failed cells (per its driver family — §2.6), and (for BYPASS phases)
populated timing/scheme-count fields. The **derived/auxiliary** outputs do
**not** carry `config.architecture_key` and are each checked differently:
- *Phase G* (`_analysis/5_3_1_fidelity.json` / `5_3_2_ranking.json`) — produced
  if its `provenance.source` points at this run's Phase A output (no new solves).
- *§5.4.1 traffic profile* (`s5_4_caselayer/caselayer_profile_*.json`) —
  self-identifies its arch via `config.hw_arch.architecture` and carries its own
  `anomalies[]`. Inspect it: the documented benign kinds `fresh_solve`,
  `zigzag_fallback`, and `f1_crosscheck_mismatch` are expected (§5.4.1); any
  other kind (e.g. `extraction_error`) is a failure.
- *§5.3.3 / §5.2.2 certs* — produced if the script ran to completion under the
  pinned `--arch` (the arch is in the command + the `on <arch>` log line).

The rerun is **complete** when all of A–G **PLUS the four easily-missed
auxiliary cert/profile steps** have produced output with the correct
architecture and no missing cells:
- **(i) §5.4.1 traffic profile** (extract_profiling — needs the L1-pkl precondition);
- **(ii) §5.3.3 chain + 7 conv anchors** (VerifyOptimalityChain, CNN);
- **(iii) §5.3.3 1×1 C=K=64 anchor** (VerifyBruteforce `--case 1x1_C64K64`);
- **(iv) §5.2.2 attention tile** (VerifyBruteforce `--case attention_tiny`, `…_TRANSFORMER`).

Steps (i), (iii), (iv) are easy to miss because an earlier version of the
§5.3.3 row was wrong — they are NOT optional (§8 E5). Nothing should be left
without an output.

### Expected results — sanity anchors (verify your reproduction against these)

The reference run produced the numbers below. Results are deterministic given
the frozen config, so a correct reproduction should land on the **same** values;
a large divergence means a config/scope error — recheck §1–§5, do not curate
around it.

| Phase | Anchor |
|---|---|
| A-EDP | MIREDO vs ZigZag total simEDP improvement: resnet18 +27.8%, vgg19bn +65.1%, alexnet +10.9%, mobilenetV2 +25.2%, EfficientNet-B0 +26.8%. MIREDO `runtime_error = 0`. |
| A-LAT | MIREDO vs ZigZag total latency speedup: resnet18 1.62×, vgg19bn 2.03×, alexnet 1.85×, mobilenetV2 1.29×, EfficientNet-B0 1.29×. |
| feas-probe (optional `MIREDO_FEAS_LOG`) | 34,333 schemes: 99.94% presolve-infeasible (Gurobi status 2), 20 prescreen cap-hits (status 9) across 6 vgg19bn layers, avg feasibility wall 5.64 s → the 20 s prescreen cap is empirically sufficient. |
| B | MIREDO vs ZigZag EDP: bert_base 1.93×, gpt2_medium_block 7.51×, tinyllama_block 1.63×; vs CiMLoop 2.85–3.09×. |
| C | 3 tradeoff points (resnet18 Latency/Energy/EDP) + `decision_comparison`; Latency cache-hits A-LAT. |
| D | 9 rows; latency degradation: `latency-only` 0%, `fixed-double-buffer` & `simplified-pipeline` = resnet18 2.16% / mobilenetV2 18.17% / EfficientNet-B0 23.05%. |
| E | 6 params × {48 sensitivity + 192 per-layer} rows, L1–L4, anomalies 0. |
| F | 4 JSONs, exit 0, anomalies 0. |
| G | §5.3.1 fidelity: latency mean/max 1.55% / 11.11%, energy 1.10% / 9.24%. §5.3.2 ranking 148/174 (85.1%), mean gap 0.10%. |
| §5.3.3 | **Latency cert clean:** 6/7 conv anchors gap 0.000% (7th = depthwise G576, no enumerable temporal ordering — structural, not a failure); 1×1 C=K=64 anchor 1409 cyc, gap 0.000%; QK^T attention tile 182 cyc, gap 0.000%. **EDP cert NOT clean** (anchor gaps 0.69–15.7%) — expected (§7). |

A's `baseline_error` anomalies (130 cosa + 130 cosa_legal across EDP+Latency) are
the CoSA depthwise/grouped (G>1) structural limit — identical every run, benign,
not a regression (§2.6).

---

## 7. Results that look weak are correct (do not "fix" by changing config)

- §5.5.1 sub-1× corner cells are **no longer "MIP budget edges"**: the root
  cause was `scout_size` starvation, now fixed by the frozen
  `scout_size = min(num_schemes, 20)`. Any residual sub-1× cell is a **real
  per-layer result**, not an artifact. Do not lower fidelity or swap configs.
- §5.3.3 is a *latency* incumbent-optimality certificate; the **EDP** certificate
  is not clean on this hardware (anchor EDP gaps 0.69–15.7%) — expected, not a
  failure. The latency cert is clean: 6/7 conv anchors gap = 0.000%, the 1×1
  anchor 1409 cyc gap = 0.000%, the attention tile 182 cyc gap = 0.000%.
- Margins over the strongest baseline are **modest by design** on this fair
  hardware. Do not switch configs to enlarge them; do not config-shop.

---

## 8. Pitfalls — do not repeat (error ledger)

Every entry below is a real pitfall hit during development. The §1–7 text
already encodes the fix; this is the consolidated checklist.

- **E1 — BYPASS over-applied to a quality phase (~9 h wasted).** A-EDP was
  launched with `MIREDO_BYPASS_MIP_CACHE=1`; it is a *quality* phase, so the 73
  within-run duplicate shapes (42 %) re-solved cold at ~7.6 min each instead of
  cache-hitting. **Fix:** BYPASS only when wall-clock/scheme-prune count is the
  reported outcome (§2.1 table). A-EDP, A-LAT, A-iso, B, C, D = no BYPASS.

- **E2 — §5.4.2 launched full-model instead of 4 case layers.** The first D
  launch used all 3 models with no `--layers` (306 cold solves) on the strength
  of guide prose that said "full models". §5.4.2 is exactly the 4 case-layer
  shapes. **Fix:** pass the 4 explicit `--layers` exactly as in the Phase D
  command above — that command is the authoritative source (§0).

- **E3 — `--methods … cosa-constrained` is an OUTPUT label, not a CLI input.**
  `RunBaselineWallTime --methods` has no `choices`; the dispatcher recognizes
  `cosa_legal` and relabels the *output* to `cosa-constrained`. Passing
  `cosa-constrained` as input silently drops the method. **Fix:** omit
  `--methods` (code default is correct) and relabel `cosa_legal → cosa-constrained`
  at curation.

- **E4 — Phase G can analyze the wrong run.** `RunPhaseGAnalysis.py` with no args
  resolves to the *newest* `output/logs_rerun_*`, which may not be the run you
  intend (e.g. a later ablation dir) — it would silently produce fidelity/ranking
  numbers off the wrong data. **Fix:** always set `MIREDO_RERUN_ROOT="$RUN"`. Same
  pin applies to `extract_profiling_caselayer.py`.

- **E5 — §5.3.3 cert row was wrong → the 1×1 anchor, attention tile, and §5.4.1
  profile were silently omitted from the first A–G pass.** The old row told you
  to get the attention tile via `VerifyOptimalityChain --architecture
  …_TRANSFORMER`, which is meaningless (VerifyOptimalityChain is CNN-only).
  **Fix:** the cert is **three** distinct `VerifyBruteforce`/`VerifyOptimalityChain`
  sub-steps (§5 cert command); the completeness check (§6) now lists all four
  auxiliary steps.

- **E6 — §5.4.1 L1-dataflow precondition not flagged.** If L1 (resnet18 Conv_8)
  is a cache hit in A-LAT and D, no `Dataflow.pkl` is written and the profiler
  fails. **Fix:** check for the pkl first; materialize with one BYPASS solve if
  absent (§5 §5.4.1 command).

- **E7 — "data gap" cried for a sanctioned scope reduction.** §5.5.2 being
  ResNet-only was twice reported as missing data; it is a **documented design
  decision**, stated in the Phase C command (§5). **Fix:** before flagging any
  number as missing, check the sanctioned scope decisions in §5 and the §6 sanity
  anchors. A stale *paper* claim → rewrite the prose; it is not a re-run trigger.
