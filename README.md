# MIREDO Developer README

This repository is the mainline implementation of the MIREDO mapping flow for IMC/CIM accelerators.
For day-to-day development, ignore the experimental scripts and generated artifacts unless you are explicitly working on them.

## What This Repository Does

Given a DNN workload in ONNX format and a hardware definition, the repository:

1. parses convolution layers from `model/<name>.onnx`
2. runs ZigZag once to build a baseline mapping/performance cache
3. converts the ZigZag result into MIREDO's internal loop/dataflow representation
4. searches candidate spatial unrolling schemes
5. builds and solves a Gurobi MIP model for each candidate
6. validates the best solution with the simulator
7. reports per-layer and whole-model latency/energy comparisons

## Read These Files First

### `run.py`

`run.py` is the top-level experiment driver.

Responsibilities:

- parse CLI arguments
- choose the workload from `model/<name>.onnx`
- call ZigZag to produce/load baseline CME results
- build the evaluation accelerator from `Architecture.<name>`
- iterate over all convolution layers
- call `SolveMapping(...)` for each layer
- compare ZigZag, solver, and simulator results
- write logs and per-layer outputs under `output/<run_name>/`

When to modify it:

- adding a new experiment entry flow
- changing CLI behavior
- changing how per-layer runs are orchestrated
- changing how baseline comparison is handled

### `SolveMapping.py`

`SolveMapping.py` is the per-layer search driver between the top-level flow and the MIP solver.

Responsibilities:

- receive one layer workload plus one accelerator instance
- enumerate candidate spatial unrolling schemes
- derive temporal unrolling from each candidate
- instantiate `utils/SolverTSS.Solver`
- run the solver
- validate candidate solutions with `Simulator/Simulax.py`
- keep the best solution for the selected objective
- save the chosen dataflow as `Dataflow.pkl`

When to modify it:

- changing search-space enumeration
- changing candidate filtering or early stopping
- changing how solver and simulator are combined
- changing best-solution selection logic

### `utils/SolverTSS.py`

`utils/SolverTSS.py` is the optimization core.

Responsibilities:

- build the Gurobi model
- define decision variables for mapping, transfer, and latency-related quantities
- derive bounds used to tighten the model
- encode feasibility and objective constraints
- solve the model and export the resulting dataflow

When to modify it:

- changing the mathematical model
- changing the latency or energy formulation
- adding/removing constraints
- debugging solver infeasibility or poor solve time

If a behavioral change affects overall experiment flow, start in `run.py`.
If it affects per-layer candidate generation, start in `SolveMapping.py`.
If it affects the actual optimization model, start in `utils/SolverTSS.py`.

## Mainline Execution Flow

The mainline path is:

`run.py`
-> parse model and args
-> get ZigZag baseline
-> convert baseline mapping to MIREDO form
-> `SolveMapping.py`
-> spatial unrolling search
-> `utils/SolverTSS.py`
-> best dataflow
-> `Simulator/Simulax.py`
-> final latency/energy report

Important support files around this flow:

- `Architecture/ZigzagAcc.py`: default hardware definition used by `run.py`
- `Architecture/ArchSpec.py`: converts ZigZag hardware objects into MIREDO's `CIM_Acc`
- `Simulator/Simulax.py`: simulator used to validate solver output
- `Evaluation/Zigzag_imc/CompatibleZigzag.py`: ZigZag-to-MIREDO conversion utilities
- `utils/UtilsFunction/OnnxParser.py`: extracts convolution loop dimensions from ONNX
- `Config/zigzag_mapping.py`: ZigZag mapping hints

## Environment Setup

The expected environment is Python 3.10 with Gurobi.

Recommended setup:

```bash
conda env create -f environment.yml
conda activate MIREDO
```

Required notes:

- `gurobipy` is required and needs a working Gurobi license
- the main flow uses the ZigZag IMC submodule under `Evaluation/Zigzag_imc/zigzag-imc` together with the CACTI wrappers in this repo
- the default working directory should be the repository root

## How To Run Experiments

### Recommended first run

Always pass `-opt` explicitly.
The parser keeps a legacy default that is not suitable for the current mainline flow.

```bash
python run.py -m resnet18 -arch ZigzagAcc -opt Latency -o dev_resnet18
```

### Other common runs

Optimize energy:

```bash
python run.py -m mobilenetV2 -arch ZigzagAcc -opt Energy -o dev_mbv2_energy
```

Optimize EDP:

```bash
python run.py -m vgg19bn -arch ZigzagAcc -opt EDP -o dev_vgg19_edp
```

Change solve time limit and MIP focus:

```bash
python run.py -m resnet50 -arch ZigzagAcc -opt Latency -t 300 -f 2 -o dev_resnet50
```

Enable verbose debug logging:

```bash
python run.py -m alexnet -arch ZigzagAcc -opt Latency --debug -o debug_alexnet
```

### Common arguments

- `-m, --model`: workload name, resolved as `model/<name>.onnx`
- `-arch, --architecture`: hardware module under `Architecture/`
- `-opt, --flag_opt`: objective, one of `Latency`, `Energy`, `EDP`
- `-t, --time`: Gurobi time limit in seconds
- `-f, --mipFocus`: Gurobi `MIPFocus`
- `-o, --outputdir`: output directory name under `output/`
- `--WS`: enable weight-stationary related flag
- `--IS`: enable input-stationary related flag
- `--NoPreSolve`: disable pre-solve search
- `--debug`: enable debug log level
- `--logger`: print logs to stdout
- `--noLogFile`: do not write the main log file

Important behavior notes:

- `--SIMU` currently acts as simulator debug/trace mode, not as the switch that enables simulation
- `--cfg` is a legacy argument and is not part of the current mainline `run.py` path

## Input And Output Conventions

### Inputs

Mainline workloads live at:

- `model/alexnet.onnx`
- `model/googlenet.onnx`
- `model/mobilenetV2.onnx`
- `model/resnet18.onnx`
- `model/resnet50.onnx`
- `model/vgg19bn.onnx`

### Outputs

A run writes to `output/<run_name>/`.

Typical contents:

- main log file
- one subdirectory per convolution layer
- `Evaluation-Layer.log` for each layer
- `Dataflow.pkl` for the selected layer mapping

ZigZag baseline caches are stored in `Evaluation/Zigzag_imc/output/`.
These are useful for reruns but are not the place to start reading the codebase.

## Suggested Developer Workflow

When you need to understand or modify behavior, use this order:

1. read `run.py` to understand the global experiment flow
2. read `SolveMapping.py` to understand per-layer search and solver invocation
3. read `utils/SolverTSS.py` to understand the optimization model
4. read `Simulator/Simulax.py` only when you need to validate execution semantics
5. read `Architecture/ZigzagAcc.py` and `Architecture/ArchSpec.py` when changing hardware assumptions

## What To Ignore At First

Unless your task explicitly targets them, do not start with:

- `tools/*.py`
- notebooks or ad hoc reports
- files under `output/`
- files under `Evaluation/Zigzag_imc/output/`
- one-off debug utilities

Those files are useful for experiments and analysis, but they are not the core path new developers should learn first.
