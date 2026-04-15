from __future__ import annotations

import sys
import os
import json
import shutil
from importlib import import_module
from pathlib import Path

import yaml
from Architecture.ArchSpec import CIM_Acc
from baseline.types import BaselineLayer
from baseline.cosa_map_parser import parse_cosa_map_16_to_baseline
from utils.GlobalUT import Logger


_MATCH_KEYS = (
    "R", "S", "P", "Q", "C", "K", "G", "B",
    "StrideH", "StrideW", "DilationH", "DilationW",
)


def _loop_key(loop_dim: dict[str, int]) -> tuple[int, ...]:
    defaults = {
        "G": 1,
        "B": 1,
        "StrideH": int(loop_dim.get("Stride", 1)),
        "StrideW": int(loop_dim.get("Stride", 1)),
        "DilationH": int(loop_dim.get("Dilation", 1)),
        "DilationW": int(loop_dim.get("Dilation", 1)),
    }
    return tuple(int(loop_dim.get(k, defaults.get(k, 0))) for k in _MATCH_KEYS)


def _loop_label(loop_dim: dict[str, int]) -> str:
    parts = [f"{k}{int(loop_dim.get(k, 1 if k == 'G' else 0))}" for k in _MATCH_KEYS]
    parts.append(f"S{int(loop_dim.get('StrideH', loop_dim.get('Stride', 1)))}x{int(loop_dim.get('StrideW', loop_dim.get('Stride', 1)))}")
    parts.append(f"D{int(loop_dim.get('DilationH', loop_dim.get('Dilation', 1)))}x{int(loop_dim.get('DilationW', loop_dim.get('Dilation', 1)))}")
    return "_".join(parts)


class CoSABaselineAdapter:
    def __init__(
        self,
        model: str,
        architecture: str,
        map_path: str | None = None,
        output_root: str | Path = "output",
    ):
        self.model = model
        self.architecture = architecture
        self._layers_by_key: dict[tuple[int, ...], BaselineLayer] = {}
        self._pending_loop_dim: dict[str, int] | None = None

        self.map_path = Path(map_path).expanduser().resolve() if map_path else None
        if self.map_path is not None:
            layers = self._load_layers(self.map_path)
            self._layers_by_key = {_loop_key(layer.loop_dim): layer for layer in layers}
            Logger.info(f"Loaded {len(layers)} CoSA baseline layer(s) from {self.map_path}")
            self._mode = "map"
            return

        self._mode = "generate"
        self._workspace = Path(output_root).expanduser().resolve() / "cosa_generated" / f"{model}_{architecture}"
        self._workspace.mkdir(parents=True, exist_ok=True)

        repo_root = Path(__file__).resolve().parents[2]
        self._cosa_root = repo_root / "112" / "Evaluation" / "CoSA" / "cosa"
        self._cosa_entry = self._cosa_root / "src" / "cosa" / "cosa.py"
        self._default_mapspace_path = self._cosa_root / "src" / "cosa" / "configs" / "mapspace" / "mapspace.yaml"
        if not self._cosa_entry.is_file():
            raise FileNotFoundError(f"CoSA entry not found: {self._cosa_entry}")
        if not self._default_mapspace_path.is_file():
            raise FileNotFoundError(f"CoSA mapspace not found: {self._default_mapspace_path}")

        self._arch_path = self._workspace / f"arch_{architecture}.yaml"
        self._mapspace_path = self._workspace / "mapspace_miredo.yaml"
        self._export_arch_yaml(self._arch_path)
        self._export_mapspace_yaml(self._mapspace_path)
        Logger.info(
            f"CoSA adapter in unified-input mode: model={model}, architecture={architecture}, "
            f"workspace={self._workspace}"
        )

    def _discover_map_files(self, path: Path) -> list[Path]:
        if path.is_file():
            if path.name != "map_16.yaml":
                raise ValueError(f"Expected map_16.yaml, got: {path}")
            return [path]
        if path.is_dir():
            files = sorted(path.rglob("map_16.yaml"))
            if not files:
                raise ValueError(f"No map_16.yaml found under directory: {path}")
            return files
        raise FileNotFoundError(path)

    @staticmethod
    def _loopdim_sidecar_path(map_file: Path) -> Path:
        return map_file.with_name("loopdim.json")

    @staticmethod
    def _write_loopdim_sidecar(map_file: Path, loop_dim: dict[str, int]) -> None:
        sidecar = CoSABaselineAdapter._loopdim_sidecar_path(map_file)
        with open(sidecar, "w", encoding="utf-8") as fp:
            json.dump(loop_dim, fp, indent=2, sort_keys=True)

    @staticmethod
    def _read_loopdim_sidecar(map_file: Path) -> dict[str, int] | None:
        sidecar = CoSABaselineAdapter._loopdim_sidecar_path(map_file)
        if not sidecar.is_file():
            return None
        with open(sidecar, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        return data if isinstance(data, dict) else None

    def _load_layers(self, path: Path) -> list[BaselineLayer]:
        layers: list[BaselineLayer] = []
        for file in self._discover_map_files(path):
            # Load sidecar first to provide source_loop_dim to IR during parsing.
            source_loop_dim = self._read_loopdim_sidecar(file)
            
            # Parse map_16.yaml and convert to baseline layer.
            layer = parse_cosa_map_16_to_baseline(file)
            
            # If sidecar was found, ensure it's propagated and update layer.loop_dim
            # to preserve original G/Stride/Dilation semantics.
            if source_loop_dim is not None:
                if hasattr(layer.raw, "source_loop_dim"):
                    layer.raw.source_loop_dim = source_loop_dim
                # Update loop_dim to use source semantics (may override inferred values)
                layer.loop_dim = dict(source_loop_dim)
            
            layers.append(layer)
        return layers

    def _build_acc(self) -> CIM_Acc:
        acc_template = import_module(f"Architecture.{self.architecture}").accelerator
        return CIM_Acc(acc_template.cores[0])

    @staticmethod
    def _bits_to_entries(capacity_bits: int, word_bits: int) -> int:
        if word_bits <= 0:
            return 1
        return max(1, int(capacity_bits) // int(word_bits))

    @staticmethod
    def _find_mem_index(acc: CIM_Acc, name: str) -> int:
        for mem in range(1, acc.Num_mem):
            if acc.mem2dict(mem) == name:
                return mem
        raise ValueError(f"Memory '{name}' not found in accelerator")

    def _export_arch_yaml(self, path: Path) -> None:
        acc = self._build_acc()

        dram_idx = self._find_mem_index(acc, "Dram")
        gb_idx = self._find_mem_index(acc, "Global_buffer")
        ib_idx = self._find_mem_index(acc, "Input_buffer")
        ob_idx = self._find_mem_index(acc, "Output_buffer")
        ir_idx = self._find_mem_index(acc, "IReg")
        or_idx = self._find_mem_index(acc, "OReg")
        macro_idx = self._find_mem_index(acc, "Macro")

        input_bits = int(acc.precision[ir_idx, 0])
        weight_bits = int(acc.precision[macro_idx, 1])
        output_bits = int(acc.precision[or_idx, 2])
        array_instances = max(1, int(acc.Num_core * acc.dimX * acc.dimY))
        core_instances = max(1, int(acc.Num_core))

        # CoSA's default hierarchy contains a dedicated WeightBuffer level.
        # ZigzagAcc does not have an independent local weight SRAM between Macro and Global_buffer,
        # so we keep this level as a placeholder (tiny + bypassed in mapspace) to preserve
        # hierarchy compatibility without inventing extra storage capacity.
        weight_buffer_entries = 1

        input_block = max(1, int(acc.bw[ib_idx] // max(1, input_bits)))
        global_block = max(1, int(acc.bw[gb_idx] // max(1, input_bits)))

        arch_obj = {
            "arch": {
                "arithmetic": {
                    "instances": array_instances,
                    "word-bits": input_bits,
                },
                "storage": [
                    {
                        "name": "Registers",
                        "entries": self._bits_to_entries(acc.memSize[macro_idx], weight_bits),
                        "instances": array_instances,
                        "word-bits": weight_bits,
                        "cluster-size": max(1, array_instances // core_instances),
                        "num-ports": 2,
                        "num-banks": 8,
                    },
                    {
                        "name": "AccumulationBuffer",
                        "entries": self._bits_to_entries(acc.memSize[ob_idx], output_bits),
                        "instances": core_instances,
                        "word-bits": output_bits,
                        "cluster-size": 1,
                        "network-word-bits": output_bits,
                        "num-ports": 2,
                        "num-banks": 1,
                    },
                    {
                        "name": "WeightBuffer",
                        "entries": weight_buffer_entries,
                        "instances": core_instances,
                        "word-bits": weight_bits,
                        "block-size": 1,
                        "num-ports": 1,
                        "num-banks": 1,
                    },
                    {
                        "name": "InputBuffer",
                        "entries": self._bits_to_entries(acc.memSize[ib_idx], input_bits),
                        "instances": core_instances,
                        "word-bits": input_bits,
                        "block-size": input_block,
                        "num-ports": 2,
                        "num-banks": 1,
                    },
                    {
                        "name": "GlobalBuffer",
                        "entries": self._bits_to_entries(acc.memSize[gb_idx], input_bits),
                        "instances": 1,
                        "word-bits": input_bits,
                        "block-size": global_block,
                        "num-ports": 2,
                        "num-banks": 1,
                    },
                    {
                        "name": "DRAM",
                        "technology": "DRAM",
                        "instances": 1,
                        "word-bits": input_bits,
                        "block-size": max(1, int(acc.bw[dram_idx] // max(1, input_bits))),
                        "bandwidth": float(acc.bw[dram_idx]),
                    },
                ],
            }
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            yaml.safe_dump(arch_obj, fp, sort_keys=False)
        Logger.info(f"Generated CoSA arch from Zigzag architecture: {path}")

    def _export_mapspace_yaml(self, path: Path) -> None:
        # Semantic mapping from ZigzagAcc memory hierarchy:
        # Weights: Dram -> Global_buffer -> Macro
        # Inputs:  Dram -> Global_buffer -> Input_buffer -> IReg
        # Outputs: OReg -> Output_buffer -> Global_buffer -> Dram
        # CoSA's WeightBuffer has no 1:1 physical level in ZigzagAcc, so it is bypassed.
        mapspace_obj = {
            "mapspace": {
                "constraints": [
                    {
                        "target": "Registers",
                        "type": "datatype",
                        "keep": ["Weights"],
                        "bypass": ["Inputs", "Outputs"],
                    },
                    {
                        "target": "AccumulationBuffer",
                        "type": "datatype",
                        "keep": ["Outputs"],
                        "bypass": ["Weights", "Inputs"],
                    },
                    {
                        "target": "WeightBuffer",
                        "type": "datatype",
                        "keep": [],
                        "bypass": ["Weights", "Inputs", "Outputs"],
                    },
                    {
                        "target": "InputBuffer",
                        "type": "datatype",
                        "keep": ["Inputs"],
                        "bypass": ["Weights", "Outputs"],
                    },
                    {
                        "target": "GlobalBuffer",
                        "type": "datatype",
                        "keep": ["Weights", "Inputs", "Outputs"],
                        "bypass": [],
                    },
                ]
            }
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            yaml.safe_dump(mapspace_obj, fp, sort_keys=False)
        Logger.info(f"Generated semantic CoSA mapspace from Zigzag architecture: {path}")

    def _export_prob_yaml(self, loop_dim: dict[str, int], path: Path) -> None:
        group = int(loop_dim.get("G", 1))
        if group != 1:
            raise ValueError(
                f"CoSA generate path currently does not support grouped convolution (G={group}). "
                "Please use a baseline map path or a model/layer with G=1."
            )

        stride_h = int(loop_dim.get("StrideH", loop_dim.get("Stride", 1)))
        stride_w = int(loop_dim.get("StrideW", loop_dim.get("Stride", 1)))
        dilation_h = int(loop_dim.get("DilationH", loop_dim.get("Dilation", 1)))
        dilation_w = int(loop_dim.get("DilationW", loop_dim.get("Dilation", 1)))
        prob_obj = {
            "problem": {
                "shape": "cnn-layer",
                "R": int(loop_dim["R"]),
                "S": int(loop_dim["S"]),
                "P": int(loop_dim["P"]),
                "Q": int(loop_dim["Q"]),
                "C": int(loop_dim["C"]),
                "K": int(loop_dim["K"]),
                "N": int(loop_dim.get("B", 1)),
                "Wstride": stride_w,
                "Hstride": stride_h,
                "Wdilation": dilation_w,
                "Hdilation": dilation_h,
            }
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            yaml.safe_dump(prob_obj, fp, sort_keys=False)

    def _store_source_loopdim(self, map_file: Path, loop_dim: dict[str, int]) -> None:
        sidecar = self._loopdim_sidecar_path(map_file)
        with open(sidecar, "w", encoding="utf-8") as fp:
            json.dump(loop_dim, fp, indent=2, sort_keys=True)

    def _ensure_cosa_runtime(self) -> None:
        cosa_src = str((self._cosa_root / "src").resolve())
        if cosa_src not in sys.path:
            sys.path.insert(0, cosa_src)

        def _prepend_env_path(var: str, value: str) -> None:
            cur = os.environ.get(var, "")
            parts = [p for p in cur.split(":") if p]
            if value in parts:
                return
            os.environ[var] = f"{value}:{cur}" if cur else value

        if not os.environ.get("COSA_DIR"):
            os.environ["COSA_DIR"] = str((self._cosa_root / "src" / "cosa").resolve())

        repo_112 = Path(__file__).resolve().parents[1]
        timeloop_roots = [
            Path(__file__).resolve().parents[2] / "timeloop",
            repo_112 / "Evaluation" / "CIMLoop" / "timeloop-accelergy-infra" / "src" / "timeloop",
            self._cosa_root.parent / "timeloop",
        ]

        # Some timeloop builds depend on boost libs from the base conda package cache.
        minconda_root = Path(sys.prefix).resolve().parents[1]
        boost_pkgs = sorted((minconda_root / "pkgs").glob("libboost-*/lib"))
        for lib_dir in boost_pkgs:
            if lib_dir.is_dir():
                _prepend_env_path("LD_LIBRARY_PATH", str(lib_dir.resolve()))

        selected_timeloop_root: Path | None = None
        for root in timeloop_roots:
            model_bin = root / "bin" / "timeloop-model"
            if model_bin.is_file():
                selected_timeloop_root = root
                _prepend_env_path("PATH", str((root / "bin").resolve()))
                if (root / "lib").is_dir():
                    _prepend_env_path("LD_LIBRARY_PATH", str((root / "lib").resolve()))
                break

        conda_lib = Path(sys.prefix) / "lib"
        if conda_lib.is_dir():
            _prepend_env_path("LD_LIBRARY_PATH", str(conda_lib.resolve()))

        if selected_timeloop_root is not None and not os.environ.get("TIMELOOP_DIR"):
            os.environ["TIMELOOP_DIR"] = str(selected_timeloop_root.resolve())

        if os.environ.get("TIMELOOP_DIR"):
            return

        timeloop_candidates = [
            self._cosa_root.parent / "timeloop",
            Path(__file__).resolve().parents[2] / "timeloop",
        ]
        for timeloop_dir in timeloop_candidates:
            if timeloop_dir.is_dir():
                os.environ["TIMELOOP_DIR"] = str(timeloop_dir.resolve())
                return

    @staticmethod
    def _derive_global_buf_idx(arch) -> int:
        preferred_names = ("GlobalBuffer", "Global_buffer", "global_buffer")
        for name in preferred_names:
            if name in arch.mem_idx:
                return int(arch.mem_idx[name])

        for name, idx in arch.mem_idx.items():
            if "global" in str(name).lower():
                return int(idx)

        # Fallback: use the level right below DRAM.
        return max(0, int(arch.mem_levels) - 2)

    @staticmethod
    def _derive_even_mapping_from_mapspace(mapspace, arch) -> tuple[list[list[int]], list[list[float]]]:
        num_vars = len(mapspace.var_idx_dict)
        num_mems = int(arch.mem_levels)

        # B[var][mem] = 1 means this variable can be kept at this memory level.
        B = [[0 for _ in range(num_mems)] for _ in range(num_vars)]
        part_ratios: list[list[float]] = []

        for mem_idx in range(num_mems):
            keep_vars: list[int] = []
            for _, var_idx in mapspace.var_idx_dict.items():
                keep = int(mapspace.bypass[mem_idx][var_idx]) != 1
                B[var_idx][mem_idx] = 1 if keep else 0
                if keep:
                    keep_vars.append(int(var_idx))

            ratios = [0.0] * num_vars
            if keep_vars:
                share = 1.0 / float(len(keep_vars))
                for var_idx in keep_vars:
                    ratios[var_idx] = share
            part_ratios.append(ratios)

        return B, part_ratios

    @staticmethod
    def _build_prefix_z_from_b(B: list[list[int]]) -> list[list[list[int]]]:
        Z: list[list[list[int]]] = []
        for var in B:
            z_var: list[list[int]] = []
            for i, val in enumerate(var):
                rank_arr = [0] * len(var)
                if int(val) == 1:
                    for j in range(i + 1):
                        rank_arr[j] = 1
                z_var.append(rank_arr)
            Z.append(z_var)
        return Z

    def _run_cosa_generate_map(self, prob_path: Path, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_cosa_runtime()

        try:
            from cosa.cosa import mip_solver
            from cosa.cosa_constants import _A
            from cosa.cosa_input_objs import Prob, Arch, Mapspace
            import cosa.run_config as run_config
        except Exception as exc:
            raise ValueError(f"Failed to import CoSA runtime in-process: {exc}") from exc

        prob = Prob(prob_path)
        arch = Arch(self._arch_path)
        mapspace = Mapspace(self._mapspace_path)
        mapspace.init(prob, arch)

        B, part_ratios = self._derive_even_mapping_from_mapspace(mapspace, arch)
        global_buf_idx = self._derive_global_buf_idx(arch)
        Z = self._build_prefix_z_from_b(B)

        Logger.info(f"CoSA adapter derived global_buf_idx={global_buf_idx}")
        Logger.info(f"CoSA adapter derived B={B}")
        Logger.info(f"CoSA adapter derived part_ratios={part_ratios}")

        prime_factors = prob.prob_factors
        strides = [prob.prob["Wstride"], prob.prob["Hstride"]]
        factor_config, spatial_config, outer_perm_config, _ = mip_solver(
            prime_factors,
            strides,
            arch,
            part_ratios,
            global_buf_idx=global_buf_idx,
            A=_A,
            Z=Z,
            compute_factor=10,
            util_factor=-0.1,
            traffic_factor=1,
        )

        update_factor_config = factor_config
        spatial_to_factor_map: dict[int, int] = {}
        idx = int(arch.mem_levels)
        for i, val in enumerate(arch.S):
            if val > 1:
                spatial_to_factor_map[i] = idx
                idx += 1

        for j, f_j in enumerate(prob.prob_factors):
            for n, _ in enumerate(f_j):
                if spatial_config[j][n] == 1:
                    mapped_level = int(factor_config[j][n])
                    update_factor_config[j][n] = spatial_to_factor_map[mapped_level]

        perm_config = mapspace.get_default_perm()
        perm_config[global_buf_idx] = outer_perm_config

        status_dict: dict = {}
        try:
            result = run_config.run_config(
                mapspace,
                None,
                perm_config,
                update_factor_config,
                status_dict,
                run_gen_map=True,
                run_gen_tc=False,
                run_sim_test=False,
                output_path=str(output_dir),
                spatial_configs=[],
                valid_check=False,
                outer_loopcount_limit=100,
            )
        except Exception as exc:
            raise ValueError(f"CoSA run_config failed during map generation: {exc}") from exc

        run_status = result.get("run_status", [0]) if isinstance(result, dict) else [0]
        if not run_status or int(run_status[0]) != 1:
            raise ValueError(f"CoSA run_config failed to generate valid map (run_status={run_status})")

        map_files = sorted(output_dir.rglob("map_16.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not map_files:
            raise ValueError(f"CoSA generated no map_16.yaml under {output_dir}.")
        if self._pending_loop_dim is not None:
            self._store_source_loopdim(map_files[0], self._pending_loop_dim)
        return map_files[0]

    def find_layer(self, loop_dim: dict[str, int]) -> BaselineLayer:
        key = _loop_key(loop_dim)
        cached = self._layers_by_key.get(key)
        if cached is not None:
            return cached

        if self._mode == "map":
            raise ValueError(
                f"No CoSA baseline layer found for loop_dim={loop_dim}. "
                f"Loaded keys={list(self._layers_by_key.keys())}"
            )

        label = _loop_label(loop_dim)
        prob_path = self._workspace / "inputs" / f"{label}.yaml"
        output_dir = self._workspace / "outputs" / label
        self._pending_loop_dim = dict(loop_dim)
        self._export_prob_yaml(loop_dim, prob_path)
        map_file = self._run_cosa_generate_map(prob_path=prob_path, output_dir=output_dir)
        Logger.info(f"Generated CoSA map for {label}: {map_file}")

        layer = parse_cosa_map_16_to_baseline(map_file)
        
        # After generation, sidecar should exist. Ensure layer.loop_dim uses source semantics.
        if hasattr(layer.raw, "source_loop_dim"):
            layer.raw.source_loop_dim = dict(loop_dim)
        # Directly use the original loop_dim to preserve G/Stride/Dilation semantics.
        layer.loop_dim = dict(loop_dim)
        
        self._layers_by_key[key] = layer
        return layer


def export_cosa_inputs_from_architecture(
    architecture: str,
    *,
    arch_out: str | Path,
    mapspace_out: str | Path,
    model: str = "resnet18",
    output_root: str | Path = "output",
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Export CoSA arch/mapspace yaml from MIREDO architecture to explicit target paths."""
    adapter = CoSABaselineAdapter(
        model=model,
        architecture=architecture,
        map_path=None,
        output_root=output_root,
    )

    src_arch = adapter._arch_path
    src_mapspace = adapter._mapspace_path
    dst_arch = Path(arch_out).expanduser().resolve()
    dst_mapspace = Path(mapspace_out).expanduser().resolve()

    dst_arch.parent.mkdir(parents=True, exist_ok=True)
    dst_mapspace.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite and dst_arch.exists():
        raise FileExistsError(f"Target arch file already exists: {dst_arch}")
    if not overwrite and dst_mapspace.exists():
        raise FileExistsError(f"Target mapspace file already exists: {dst_mapspace}")

    shutil.copyfile(src_arch, dst_arch)
    shutil.copyfile(src_mapspace, dst_mapspace)

    Logger.info(
        f"Exported CoSA inputs from architecture={architecture}: "
        f"arch={dst_arch}, mapspace={dst_mapspace}"
    )
    return dst_arch, dst_mapspace
