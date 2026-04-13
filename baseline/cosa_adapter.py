from __future__ import annotations

from copy import deepcopy
from importlib import import_module
from pathlib import Path

import yaml
from Architecture.ArchSpec import CIM_Acc
from baseline.types import BaselineLayer
from baseline.cosa_map_parser import parse_cosa_map_16_to_baseline
from utils.GlobalUT import Logger
from utils.CoSAUtils import (
    assert_cosa_local_ready,
    cosa_default_mapspace_path,
    cosa_entry_file,
    cosa_submodule_root,
    run_cosa,
)


_MATCH_KEYS = ("R", "S", "P", "Q", "C", "K", "G", "Stride")


def _loop_key(loop_dim: dict[str, int]) -> tuple[int, ...]:
    return tuple(int(loop_dim.get(k, 1 if k in ("G", "Stride") else 0)) for k in _MATCH_KEYS)


def _loop_label(loop_dim: dict[str, int]) -> str:
    parts = [f"{k}{int(loop_dim.get(k, 1 if k in ('G', 'Stride') else 0))}" for k in _MATCH_KEYS]
    return "_".join(parts)


def _same_core_layer(a: dict[str, int], b: dict[str, int]) -> bool:
    for key in ("R", "S", "P", "Q", "C", "K"):
        if int(a.get(key, 0)) != int(b.get(key, 0)):
            return False
    return int(a.get("G", 1)) == int(b.get("G", 1))


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

        self._cosa_root = cosa_submodule_root()
        self._cosa_entry = cosa_entry_file()
        self._default_mapspace_path = cosa_default_mapspace_path()
        if not self._cosa_entry.is_file():
            raise FileNotFoundError(f"CoSA entry not found: {self._cosa_entry}")
        if not self._default_mapspace_path.is_file():
            raise FileNotFoundError(f"CoSA mapspace not found: {self._default_mapspace_path}")
        assert_cosa_local_ready()

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
            if path.suffix.lower() not in (".yaml", ".yml"):
                raise ValueError(f"Unsupported CoSA map file type: {path}")
            return [path]
        if path.is_dir():
            files = sorted(path.rglob("map_16.yaml"))
            if not files:
                files = sorted(list(path.rglob("*.yaml")) + list(path.rglob("*.yml")))
            if not files:
                raise ValueError(f"No CoSA yaml file found under directory: {path}")
            return files
        raise FileNotFoundError(path)

    def _load_layers(self, path: Path) -> list[BaselineLayer]:
        return [parse_cosa_map_16_to_baseline(file) for file in self._discover_map_files(path)]

    @staticmethod
    def _with_requested_loop_dim(layer: BaselineLayer, loop_dim: dict[str, int]) -> BaselineLayer:
        if not _same_core_layer(layer.loop_dim, loop_dim):
            raise ValueError(
                "CoSA mapping/workload mismatch: "
                f"map loop_dim={layer.loop_dim}, requested loop_dim={loop_dim}"
            )
        result = deepcopy(layer)
        result.loop_dim = {
            key: int(value) if value is not None else value
            for key, value in loop_dim.items()
        }
        result.loop_dim.setdefault("G", 1)
        result.loop_dim.setdefault("Stride", 1)
        result.loop_dim.setdefault("Padding", 0)
        return result

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
        if path.is_file():
            return
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
                        "num-banks": 1,
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
                        "entries": self._bits_to_entries(acc.memSize[gb_idx], weight_bits),
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
                        "block-size": 1,
                        "num-ports": 2,
                        "num-banks": 1,
                    },
                    {
                        "name": "GlobalBuffer",
                        "entries": self._bits_to_entries(acc.memSize[gb_idx], input_bits),
                        "instances": 1,
                        "word-bits": input_bits,
                        "block-size": 1,
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
        if path.is_file():
            return

        with open(self._default_mapspace_path, "r", encoding="utf-8") as fp:
            mapspace_obj = yaml.safe_load(fp)

        constraints = mapspace_obj.get("mapspace", {}).get("constraints", [])
        for constraint in constraints:
            if constraint.get("type") != "datatype":
                continue
            if constraint.get("target") != "GlobalBuffer":
                continue

            # MIREDO 的 Global_buffer 是共享位宽容量。为避免 CoSA 在该层对 I/O 双 keep
            # 造成大规模容量不匹配，这里约束为仅 keep Inputs。
            constraint["keep"] = ["Inputs"]
            constraint["bypass"] = ["Weights", "Outputs"]

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            yaml.safe_dump(mapspace_obj, fp, sort_keys=False)
        Logger.info(f"Generated MIREDO-compatible CoSA mapspace: {path}")

    def _export_prob_yaml(self, loop_dim: dict[str, int], path: Path) -> None:
        if int(loop_dim.get("G", 1)) != 1:
            raise ValueError(
                "CoSA baseline cannot export grouped/depthwise convolution layers fairly: "
                f"expected G=1 but got G={loop_dim.get('G')} for loop_dim={loop_dim}. "
                "The public CoSA interface used here has loop dimensions R,S,P,Q,C,K,N "
                "and this adapter does not implement a validated group-conv decomposition."
            )
        stride = int(loop_dim.get("Stride", 1))
        dilation = int(loop_dim.get("Dilation", 1))
        prob_obj = {
            "problem": {
                "shape": "cnn-layer",
                "R": int(loop_dim["R"]),
                "S": int(loop_dim["S"]),
                "P": int(loop_dim["P"]),
                "Q": int(loop_dim["Q"]),
                "C": int(loop_dim["C"]),
                "K": int(loop_dim["K"]),
                "N": int(loop_dim.get("B") or 1),
                "Wstride": stride,
                "Hstride": stride,
                "Wdilation": dilation,
                "Hdilation": dilation,
            }
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            yaml.safe_dump(prob_obj, fp, sort_keys=False)

    def _run_cosa_generate_map(self, prob_path: Path, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        proc = run_cosa(
            prob_path=prob_path,
            arch_path=self._arch_path,
            mapspace_path=self._mapspace_path,
            output_dir=output_dir,
        )
        if proc.returncode != 0:
            tail = "\n".join(proc.stdout.splitlines()[-80:])
            raise ValueError(f"CoSA run failed (exit={proc.returncode}).\n{tail}")

        map_files = sorted(output_dir.rglob("map_16.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not map_files:
            tail = "\n".join(proc.stdout.splitlines()[-80:])
            raise ValueError(f"CoSA generated no map_16.yaml under {output_dir}.\n{tail}")
        return map_files[0]

    def find_layer(self, loop_dim: dict[str, int]) -> BaselineLayer:
        key = _loop_key(loop_dim)
        cached = self._layers_by_key.get(key)
        if cached is not None:
            return self._with_requested_loop_dim(cached, loop_dim)

        for layer in self._layers_by_key.values():
            if _same_core_layer(layer.loop_dim, loop_dim):
                updated = self._with_requested_loop_dim(layer, loop_dim)
                self._layers_by_key[key] = updated
                return updated

        if self._mode == "map":
            raise ValueError(
                f"No CoSA baseline layer found for loop_dim={loop_dim}. "
                f"Loaded keys={list(self._layers_by_key.keys())}"
            )

        label = _loop_label(loop_dim)
        prob_path = self._workspace / "inputs" / f"{label}.yaml"
        output_dir = self._workspace / "outputs" / label
        self._export_prob_yaml(loop_dim, prob_path)
        map_file = self._run_cosa_generate_map(prob_path=prob_path, output_dir=output_dir)
        Logger.info(f"Generated CoSA map for {label}: {map_file}")

        layer = parse_cosa_map_16_to_baseline(map_file)
        layer = self._with_requested_loop_dim(layer, loop_dim)
        self._layers_by_key[key] = layer
        return layer
