from __future__ import annotations

import subprocess
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

    def _run_cosa_generate_map(self, prob_path: Path, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "cosa.cosa",
            "-o",
            str(output_dir),
            "-ap",
            str(self._arch_path),
            "-mp",
            str(self._mapspace_path),
            "-pp",
            str(prob_path),
        ]
        env = os.environ.copy()
        cosa_src = str(self._cosa_root / "src")
        env["PYTHONPATH"] = f"{cosa_src}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else cosa_src
        if not env.get("TIMELOOP_DIR"):
            timeloop_candidates = [
                self._cosa_root.parent / "timeloop",
                Path(__file__).resolve().parents[2] / "timeloop",
            ]
            for timeloop_dir in timeloop_candidates:
                if timeloop_dir.is_dir():
                    env["TIMELOOP_DIR"] = str(timeloop_dir.resolve())
                    break
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
        )
        if proc.returncode != 0:
            tail = "\n".join(proc.stdout.splitlines()[-80:])
            raise ValueError(f"CoSA run failed (exit={proc.returncode}).\n{tail}")

        map_files = sorted(output_dir.rglob("map_16.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not map_files:
            tail = "\n".join(proc.stdout.splitlines()[-80:])
            raise ValueError(f"CoSA generated no map_16.yaml under {output_dir}.\n{tail}")
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
