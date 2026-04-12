from __future__ import annotations

from pathlib import Path
import json
import os
import re
import subprocess
import sys
import yaml

from baseline.cimloop_map_parser import parse_cimloop_baseline_yaml
from baseline.cimloop_hw_bridge import CimloopHardwareBridge
from baseline.types import BaselineLayer
from utils.GlobalUT import Logger


_MATCH_KEYS = ("R", "S", "P", "Q", "C", "K", "G")
_OP_NAME_TO_KEY = {"Inputs": "I", "Weights": "W", "Outputs": "O"}
_DIM_ALIAS = {"M": "K"}
_ALLOWED_DIMS = {"R", "S", "P", "Q", "C", "K", "G", "M"}
_DEFAULT_HIST = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]


def _loop_key(loop_dim: dict[str, int]) -> tuple[int, ...]:
    return tuple(int(loop_dim.get(k, 1 if k == "G" else 0)) for k in _MATCH_KEYS)


def _loop_label(loop_dim: dict[str, int]) -> str:
    parts = [f"{k}{int(loop_dim.get(k, 1 if k == 'G' else 0))}" for k in _MATCH_KEYS]
    parts.append(f"S{int(loop_dim.get('Stride', 1))}")
    parts.append(f"D{int(loop_dim.get('Dilation', 1))}")
    return "_".join(parts)


class CimloopBaselineAdapter:
    def __init__(
        self,
        model: str,
        architecture: str,
        map_path: str | None = None,
        output_root: str | Path = "output",
        macro: str | None = None,
        system: str | None = None,
        tile: str | None = None,
        chip: str | None = None,
        iso: str | None = None,
        hardware_from_arch: bool = True,
    ):
        self.model = model
        self.architecture = architecture
        self.macro = macro
        self.system = system
        self.tile = tile
        self.chip = chip
        self.iso = iso
        self.hardware_from_arch = hardware_from_arch
        self._hardware_variables: dict[str, int | float | bool] = {"CIM_ARCHITECTURE": True}
        self._default_hw_profile = {
            "macro": "isaac_isca_2016",
            "system": "ws_dummy_buffer_one_macro",
            "tile": None,
            "chip": None,
            "iso": None,
            "variables": {"CIM_ARCHITECTURE": True},
        }
        self._compatibility_hw_profiles = [
            {
                "name": "albireo_capacity_fallback",
                "macro": "albireo_isca_2021",
                "system": "ws_dummy_buffer_one_macro",
                "tile": None,
                "chip": None,
                "iso": None,
                "variables": {"CIM_ARCHITECTURE": True},
            }
        ]
        self._arch_hw_profile_active = False
        self.map_path = Path(map_path).expanduser().resolve() if map_path else None
        self._layers_by_key: dict[tuple[int, ...], BaselineLayer] = {}

        if self.map_path is not None:
            layers = self._load_layers(self.map_path)
            if len(layers) == 0:
                raise ValueError(f"No valid CIMLoop baseline layer found under: {self.map_path}")

            self._layers_by_key = {_loop_key(layer.loop_dim): layer for layer in layers}
            self._mode = "map"
            Logger.info(f"Loaded {len(layers)} CIMLoop baseline layer(s) from {self.map_path}")
            return

        self._mode = "generate"
        self._workspace = Path(output_root).expanduser().resolve() / "cimloop_generated" / f"{model}_{architecture}"
        self._input_workload_dir = self._workspace / "inputs" / "workloads"
        self._output_root = self._workspace / "outputs"
        self._repo_root = Path(__file__).resolve().parents[1]
        self._cimloop_root = self._repo_root / "Evaluation" / "CIMLoop" / "cimloop"
        self._cimloop_utils = self._cimloop_root / "workspace" / "scripts" / "utils.py"
        self._docker_workspace = self._cimloop_root / "workspace"
        self._docker_generated_workload_dir = (
            self._docker_workspace / "models" / "workloads" / "__miredo_generated__"
        )

        if self.hardware_from_arch:
            try:
                hw_spec = CimloopHardwareBridge(
                    architecture=self.architecture,
                    cimloop_root=self._cimloop_root,
                    base_macro=(self.macro or "isaac_isca_2016"),
                ).build()
                self.macro = self.macro or hw_spec.macro
                self.system = self.system or hw_spec.system
                self.tile = self.tile if self.tile is not None else hw_spec.tile
                self.chip = self.chip if self.chip is not None else hw_spec.chip
                self.iso = self.iso or hw_spec.iso
                self._hardware_variables.update(hw_spec.variables)
                self._arch_hw_profile_active = True
                Logger.info(
                    f"CIMLoop hardware profile from {hw_spec.source}: "
                    f"macro={self.macro}, system={self.system}, tile={self.tile}, chip={self.chip}, iso={self.iso}"
                )
            except Exception as exc:  # noqa: BLE001
                Logger.warning(
                    "Failed to build CIMLoop hardware profile from MIREDO architecture; "
                    f"fallback to manual/default hardware parameters. reason={exc}"
                )

        self.macro = self.macro or "isaac_isca_2016"
        self.system = self.system or "ws_dummy_buffer_one_macro"
        self._input_workload_dir.mkdir(parents=True, exist_ok=True)
        self._output_root.mkdir(parents=True, exist_ok=True)
        if not self._cimloop_utils.is_file():
            raise FileNotFoundError(f"CIMLoop runner entry not found: {self._cimloop_utils}")
        Logger.info(
            f"CIMLoop adapter in unified-input mode: model={model}, architecture={architecture}, "
            f"workspace={self._workspace}"
        )

    def _activate_default_hw_profile(self) -> None:
        self.macro = self._default_hw_profile["macro"]
        self.system = self._default_hw_profile["system"]
        self.tile = self._default_hw_profile["tile"]
        self.chip = self._default_hw_profile["chip"]
        self.iso = self._default_hw_profile["iso"]
        self._hardware_variables = dict(self._default_hw_profile["variables"])
        self._arch_hw_profile_active = False

    def _snapshot_hw_profile(self) -> dict:
        return {
            "macro": self.macro,
            "system": self.system,
            "tile": self.tile,
            "chip": self.chip,
            "iso": self.iso,
            "variables": dict(self._hardware_variables),
            "arch_active": self._arch_hw_profile_active,
        }

    def _restore_hw_profile(self, snapshot: dict) -> None:
        self.macro = snapshot["macro"]
        self.system = snapshot["system"]
        self.tile = snapshot["tile"]
        self.chip = snapshot["chip"]
        self.iso = snapshot["iso"]
        self._hardware_variables = dict(snapshot["variables"])
        self._arch_hw_profile_active = bool(snapshot["arch_active"])

    @staticmethod
    def _is_mapper_infeasible_output(stdout_text: str) -> bool:
        return (
            "no valid mappings found within search criteria" in stdout_text
            or "Could not find cycles in stats" in stdout_text
        )

    @staticmethod
    def _should_fallback_to_docker(local_stdout: str) -> tuple[bool, str]:
        if "No module named 'pytimeloop'" in local_stdout:
            return True, "Host environment misses pytimeloop"
        if "Permission denied" in local_stdout and "workspace/outputs" in local_stdout:
            return True, "Host CIMLoop workspace outputs directory is not writable"
        if CimloopBaselineAdapter._is_mapper_infeasible_output(local_stdout):
            return True, "Host timeloop-mapper found no valid mappings"
        if (
            "Timeloop mapper failed with return code -6" in local_stdout
            or "problem-shape.cpp" in local_stdout
            or "term.isArray" in local_stdout
        ):
            return True, "Host timeloop-mapper aborted on workload parsing"
        return False, ""

    def _apply_hw_profile(self, profile: dict, *, arch_active: bool) -> None:
        self.macro = profile["macro"]
        self.system = profile["system"]
        self.tile = profile["tile"]
        self.chip = profile["chip"]
        self.iso = profile["iso"]
        self._hardware_variables = dict(profile["variables"])
        self._arch_hw_profile_active = arch_active

    @staticmethod
    def _docker_compose_env() -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("DOCKER_ARCH", "amd64")
        return env

    def _ensure_docker_tutorial_up(self) -> None:
        env = self._docker_compose_env()

        ps = subprocess.run(
            ["docker", "compose", "ps", "--status", "running", "-q", "tutorial"],
            cwd=str(self._cimloop_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if ps.returncode == 0 and len(ps.stdout.strip()) > 0:
            return

        up = subprocess.run(
            ["docker", "compose", "up", "-d", "tutorial"],
            cwd=str(self._cimloop_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if up.returncode != 0:
            tail = "\n".join(up.stdout.splitlines()[-120:])
            raise ValueError(f"Failed to start CIMLoop docker tutorial service.\n{tail}")

        ps_after = subprocess.run(
            ["docker", "compose", "ps", "--status", "running", "-q", "tutorial"],
            cwd=str(self._cimloop_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if ps_after.returncode != 0 or len(ps_after.stdout.strip()) == 0:
            tail = "\n".join(ps_after.stdout.splitlines()[-120:])
            raise ValueError(f"CIMLoop docker tutorial service is not running after startup attempt.\n{tail}")

    def _run_cimloop_generate_layer_local(self, prob_path: Path, result_json: Path) -> tuple[int, str]:
        variables_literal = repr(self._hardware_variables)
        cmd = [
            sys.executable,
            "-c",
            (
                "import importlib.util, json, pathlib\n"
                "import pytimeloop.timeloopfe.v4.output_parsing as _op\n"
                f"utils_path = pathlib.Path(r'{self._cimloop_utils}')\n"
                "spec = importlib.util.spec_from_file_location('cimloop_utils', utils_path)\n"
                "mod = importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(mod)\n"
                "_orig_get_area = _op.get_area_from_art\n"
                "def _safe_get_area(path):\n"
                "    try:\n"
                "        return _orig_get_area(path)\n"
                "    except FileNotFoundError:\n"
                "        return {'__missing_art__': 1e-12}\n"
                "_op.get_area_from_art = _safe_get_area\n"
                "res = mod.run_layer("
                f"macro={repr(self.macro)}, "
                f"layer={repr(str(prob_path))}, "
                f"iso={repr(self.iso)}, "
                f"tile={repr(self.tile)}, "
                f"chip={repr(self.chip)}, "
                f"system={repr(self.system)}, "
                f"variables={variables_literal}"
                ")\n"
                "out = {'mapping': getattr(res, 'mapping', None)}\n"
                f"pathlib.Path(r'{result_json}').write_text(json.dumps(out), encoding='utf-8')\n"
            ),
        ]

        env = os.environ.copy()
        workspace_dir = str((self._cimloop_root / "workspace").resolve())
        env["PYTHONPATH"] = f"{workspace_dir}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else workspace_dir

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            cwd=str(self._repo_root),
            env=env,
        )
        return proc.returncode, proc.stdout

    def _run_cimloop_generate_layer_docker(self, layer_arg: str, workload_obj: dict | None) -> tuple[int, str]:
        variables_literal = repr(self._hardware_variables)
        preload = ""
        if workload_obj is not None:
            layer_label = layer_arg.split("/")[-1]
            workload_literal = repr(workload_obj)
            preload = (
                f"workload = {workload_literal}\n"
                f"workload_path = pathlib.Path('/home/workspace/models/workloads/__miredo_generated__/{layer_label}.yaml')\n"
                "workload_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "workload_path.write_text(yaml.safe_dump(workload, sort_keys=False), encoding='utf-8')\n"
            )

        cmd = [
            "docker",
            "compose",
            "run",
            "--rm",
            "-T",
            "tutorial",
            "python3",
            "-c",
            (
                "import importlib.util, json, pathlib, yaml\n"
                "utils_path = pathlib.Path('/home/workspace/scripts/utils.py')\n"
                "spec = importlib.util.spec_from_file_location('cimloop_utils', utils_path)\n"
                "mod = importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(mod)\n"
                f"{preload}"
                "res = mod.run_layer("
                f"macro={repr(self.macro)}, "
                f"layer={repr(layer_arg)}, "
                f"iso={repr(self.iso)}, "
                f"tile={repr(self.tile)}, "
                f"chip={repr(self.chip)}, "
                f"system={repr(self.system)}, "
                f"variables={variables_literal}"
                ")\n"
                "out = {'mapping': getattr(res, 'mapping', None)}\n"
                "print('__MIREDO_MAPPING_JSON_BEGIN__')\n"
                "print(json.dumps(out))\n"
                "print('__MIREDO_MAPPING_JSON_END__')\n"
            ),
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            cwd=str(self._cimloop_root),
            env=self._docker_compose_env(),
        )
        return proc.returncode, proc.stdout

    @staticmethod
    def _extract_docker_mapping_payload(stdout_text: str) -> dict | None:
        marker_begin = "__MIREDO_MAPPING_JSON_BEGIN__"
        marker_end = "__MIREDO_MAPPING_JSON_END__"
        if marker_begin not in stdout_text or marker_end not in stdout_text:
            return None

        body = stdout_text.split(marker_begin, 1)[1].split(marker_end, 1)[0].strip()
        if len(body) == 0:
            return None
        try:
            payload = json.loads(body)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _discover_files(self, path: Path) -> list[Path]:
        if path.is_file():
            if path.suffix.lower() not in (".yaml", ".yml"):
                raise ValueError(f"Unsupported CIMLoop baseline file type: {path}")
            return [path]

        if not path.is_dir():
            raise FileNotFoundError(path)

        patterns = (
            "**/map_16.yaml",
            "**/*cimloop*map*.yaml",
            "**/*cimloop*baseline*.yaml",
            "**/*baseline_layer*.yaml",
            "**/*mapping_result*.yaml",
            "**/*cimloop*map*.yml",
            "**/*cimloop*baseline*.yml",
            "**/*baseline_layer*.yml",
            "**/*mapping_result*.yml",
        )

        files: list[Path] = []
        for pattern in patterns:
            files.extend(path.rglob(pattern.replace("**/", "")))

        uniq = sorted(set(files))
        if len(uniq) > 0:
            return uniq

        # Fallback for manually curated directories.
        fallback = sorted(list(path.rglob("*.yaml")) + list(path.rglob("*.yml")))
        if len(fallback) == 0:
            raise ValueError(f"No yaml file found under directory: {path}")
        return fallback

    def _load_layers(self, path: Path) -> list[BaselineLayer]:
        layers: list[BaselineLayer] = []
        errors: list[str] = []

        for file in self._discover_files(path):
            try:
                layer = parse_cimloop_baseline_yaml(file)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{file}: {exc}")
                continue
            layers.append(layer)

        if len(layers) == 0 and len(errors) > 0:
            detail = "\n".join(errors[:20])
            raise ValueError(f"Failed to parse CIMLoop baseline yaml files under {path}:\n{detail}")

        if len(errors) > 0:
            Logger.warning(
                f"Skipped {len(errors)} non-baseline yaml file(s) when loading CIMLoop baseline from {path}"
            )
        return layers

    def _export_prob_yaml(self, loop_dim: dict[str, int], path: Path) -> None:
        # Keep the schema aligned with CiMLoop workload conventions (problem.instance).
        stride = int(loop_dim.get("Stride", 1))
        dilation = int(loop_dim.get("Dilation", 1))
        r = int(loop_dim["R"])
        s = int(loop_dim["S"])
        p = int(loop_dim["P"])
        q = int(loop_dim["Q"])

        shape_obj = {
            "coefficients": [
                {"name": "Wstride", "default": 1},
                {"name": "Hstride", "default": 1},
                {"name": "Wdilation", "default": 1},
                {"name": "Hdilation", "default": 1},
            ],
            "data_spaces": [
                {
                    "name": "Weights",
                    "projection": [[["C"]], [["M"]], [["R"]], [["S"]], [["Y"]], [["G"]]],
                },
                {
                    "name": "Inputs",
                    "projection": [
                        [["N"]],
                        [["C"]],
                        [["R", "Wdilation"], ["P", "Wstride"]],
                        [["S", "Hdilation"], ["Q", "Hstride"]],
                        [["X"]],
                        [["G"]],
                    ],
                },
                {
                    "name": "Outputs",
                    "projection": [[["N"]], [["M"]], [["Q"]], [["P"]], [["Z"]], [["G"]]],
                    "read_write": True,
                },
            ],
            "dimensions": ["C", "M", "R", "S", "N", "P", "Q", "X", "Y", "Z", "G"],
        }

        instance_obj = {
            "N": int(loop_dim.get("B", 1)),
            "X": int(loop_dim.get("InputBits", 16)),
            "G": int(loop_dim.get("G", 1)),
            "C": int(loop_dim["C"]),
            "Y": int(loop_dim.get("WeightBits", 16)),
            "M": int(loop_dim["K"]),
            "P": p,
            "Q": q,
            "R": r,
            "S": s,
            "Z": int(loop_dim.get("OutputBits", 16)),
        }
        if stride != 1:
            instance_obj["Hstride"] = stride
            instance_obj["Wstride"] = stride
        if dilation != 1:
            instance_obj["Hdilation"] = dilation
            instance_obj["Wdilation"] = dilation

        workload_obj = {
            "problem": {
                "version": 0.4,
                "instance": instance_obj,
                "shape": shape_obj,
                "name": "Conv2d",
                "dnn_name": self.model,
                "notes": "Generated by MIREDO CimloopBaselineAdapter",
                "histograms": {
                    "Inputs": _DEFAULT_HIST,
                    "Weights": _DEFAULT_HIST,
                    "Outputs": _DEFAULT_HIST,
                },
            }
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            yaml.safe_dump(workload_obj, fp, sort_keys=False)

    def _discover_generated_layer_files(self, layer_output_dir: Path) -> list[Path]:
        if not layer_output_dir.is_dir():
            return []

        patterns = (
            "baseline_layer.yaml",
            "baseline_layer.yml",
            "mapping_result.yaml",
            "mapping_result.yml",
            "map_16.yaml",
            "map_16.yml",
            "*.baseline.yaml",
            "*.baseline.yml",
        )

        files: list[Path] = []
        for p in patterns:
            files.extend(layer_output_dir.rglob(p))
        return sorted(set(files))

    @staticmethod
    def _parse_inline_instance(workload_text: str) -> dict[str, int] | None:
        m = re.search(r"instance\s*:\s*\{([^}]*)\}", workload_text)
        if m is None:
            return None

        fields: dict[str, int] = {}
        for token in m.group(1).split(","):
            pair = token.strip()
            if len(pair) == 0 or ":" not in pair:
                continue
            key, raw = pair.split(":", 1)
            k = key.strip()
            v = raw.strip()
            try:
                fields[k] = int(float(v))
            except ValueError:
                continue
        return fields if len(fields) > 0 else None

    def _find_builtin_layer_arg(self, loop_dim: dict[str, int]) -> str | None:
        model_dir = self._cimloop_root / "workspace" / "models" / "workloads" / self.model
        if not model_dir.is_dir():
            return None

        required = {
            "C": int(loop_dim["C"]),
            "M": int(loop_dim["K"]),
            "P": int(loop_dim["P"]),
            "Q": int(loop_dim["Q"]),
            "R": int(loop_dim["R"]),
            "S": int(loop_dim["S"]),
            "G": int(loop_dim.get("G", 1)),
        }
        target_stride = int(loop_dim.get("Stride", 1))
        target_dilation = int(loop_dim.get("Dilation", 1))

        for wf in sorted(model_dir.glob("*.yaml")):
            try:
                text = wf.read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001
                continue

            fields = self._parse_inline_instance(text)
            if fields is None:
                continue

            mismatch = False
            for k, v in required.items():
                default_val = 1 if k in ("R", "S", "G") else -1
                actual = int(fields.get(k, default_val))
                if actual != v:
                    mismatch = True
                    break
            if mismatch:
                continue

            stride = int(fields.get("Hstride", fields.get("HStride", 1)))
            dilation = int(fields.get("Hdilation", 1))
            if stride != target_stride or dilation != target_dilation:
                continue

            return f"{self.model}/{wf.stem}"

        return None

    @staticmethod
    def _norm_dim_name(dim: str) -> str:
        base = _DIM_ALIAS.get(dim, dim)
        return base

    @staticmethod
    def _split_loop_tokens(mapping_text: str) -> list[dict]:
        sections: list[dict] = []
        current = None

        header_re = re.compile(r"^([A-Za-z0-9_]+)\s*\[(.*?)\]\s*$")
        # Support both "[0:32)" and "[0:29,24)" loop forms emitted by timeloop mapper.
        loop_re = re.compile(r"for\s+([A-Za-z]+)\s+in\s+\[0:(\d+)(?:,\d+)?\)")

        for raw in mapping_text.splitlines():
            line = raw.strip()
            if not line:
                continue

            m_header = header_re.match(line)
            if m_header is not None:
                name = m_header.group(1)
                payload = m_header.group(2)
                operands = []
                for op_name, op_key in _OP_NAME_TO_KEY.items():
                    if re.search(rf"\b{op_name}:", payload):
                        operands.append(op_key)

                current = {
                    "name": name,
                    "operands": operands,
                    "temporal": [],
                    "spatial": [],
                }
                sections.append(current)
                continue

            if current is None:
                continue
            if "for " not in line:
                continue

            m_loop = loop_re.search(line)
            if m_loop is None:
                continue

            dim = m_loop.group(1)
            if dim not in _ALLOWED_DIMS:
                continue

            dim = CimloopBaselineAdapter._norm_dim_name(dim)
            size = int(m_loop.group(2))
            if size <= 1:
                continue

            target = "spatial" if "(Spatial" in line or "(spatial" in line else "temporal"
            current[target].append((dim, size))

        return sections

    @staticmethod
    def _build_baseline_yaml_payload(loop_dim: dict[str, int], sections: list[dict]) -> dict:
        temporal_mapping: dict[str, list[list[list]]]= {"I": [], "W": [], "O": []}
        spatial_mapping: dict[str, list[list[list]]]= {"I": [], "W": [], "O": []}

        # Timeloop map text contains many interconnect sections without explicit operand tags.
        # Keep those loops as one merged global level so replay can preserve key factors (R/S/C/K/P/Q).
        global_temporal: list[list] = []
        global_spatial: list[list] = []
        for sec in sections:
            if len(sec["operands"]) != 0:
                continue
            for dim, size in sec["temporal"]:
                global_temporal.append([dim, int(size)])
            for dim, size in sec["spatial"]:
                global_spatial.append([dim, int(size)])

        for op in ("I", "W", "O"):
            path = [sec for sec in sections if op in sec["operands"] and (len(sec["temporal"]) > 0 or len(sec["spatial"]) > 0)]

            temporal_layers = [
                [[dim, int(size)] for dim, size in sec["temporal"]]
                for sec in path
                if len(sec["temporal"]) > 0
            ]
            spatial_layers = [
                [[dim, int(size)] for dim, size in sec["spatial"]]
                for sec in path
                if len(sec["spatial"]) > 0
            ]

            if len(global_temporal) > 0:
                temporal_layers.append([item[:] for item in global_temporal])
            if len(global_spatial) > 0:
                spatial_layers.append([item[:] for item in global_spatial])

            temporal_mapping[op] = temporal_layers
            spatial_mapping[op] = spatial_layers

        double_buffer_flag = {
            op: [False] + [False] * len(temporal_mapping[op])
            for op in ("I", "W", "O")
        }
        top_r_loop_size = {
            op: [1] * (len(temporal_mapping[op]) + 1)
            for op in ("I", "W", "O")
        }

        loop_dim_out = {
            "R": int(loop_dim["R"]),
            "S": int(loop_dim["S"]),
            "P": int(loop_dim["P"]),
            "Q": int(loop_dim["Q"]),
            "C": int(loop_dim["C"]),
            "K": int(loop_dim["K"]),
            "G": int(loop_dim.get("G", 1)),
        }
        for key in ("B", "H", "W", "Stride", "Padding", "Dilation"):
            if key in loop_dim:
                loop_dim_out[key] = int(loop_dim[key])

        return {
            "loop_dim": loop_dim_out,
            "temporal_mapping": temporal_mapping,
            "spatial_mapping": spatial_mapping,
            "double_buffer_flag": double_buffer_flag,
            "top_r_loop_size": top_r_loop_size,
        }

    def _run_cimloop_generate_layer(self, loop_dim: dict[str, int], prob_path: Path, layer_output_dir: Path) -> None:
        layer_output_dir.mkdir(parents=True, exist_ok=True)
        result_json = layer_output_dir / "cimloop_result.json"
        local_prob_path = prob_path
        docker_layer_arg = self._find_builtin_layer_arg(loop_dim)
        if docker_layer_arg is not None:
            builtin_prob_path = self._cimloop_root / "workspace" / "models" / "workloads" / f"{docker_layer_arg}.yaml"
            if builtin_prob_path.is_file():
                local_prob_path = builtin_prob_path
                Logger.info(f"Using built-in CIMLoop workload for local generation: {docker_layer_arg}")

        local_outputs_dir = self._cimloop_root / "workspace" / "outputs"
        if local_outputs_dir.is_dir() and not os.access(local_outputs_dir, os.W_OK):
            local_rc = 1
            local_out = (
                "Permission denied: local CIMLoop workspace/outputs is not writable "
                f"({local_outputs_dir})"
            )
        else:
            local_rc, local_out = self._run_cimloop_generate_layer_local(prob_path=local_prob_path, result_json=result_json)
        use_docker, fallback_reason = self._should_fallback_to_docker(local_out)

        if local_rc != 0 and use_docker:
            layer_label = prob_path.stem
            workload_obj = None
            if docker_layer_arg is None:
                workload_obj = yaml.safe_load(prob_path.read_text(encoding="utf-8"))
                docker_layer_arg = f"__miredo_generated__/{layer_label}"
            else:
                Logger.info(f"Using built-in CIMLoop workload for docker generation: {docker_layer_arg}")

            Logger.warning(
                f"{fallback_reason}; falling back to CIMLoop docker service for generation"
            )
            original_hw_profile = self._snapshot_hw_profile()
            docker_rc, docker_out = self._run_cimloop_generate_layer_docker(
                layer_arg=docker_layer_arg,
                workload_obj=workload_obj,
            )
            if docker_rc != 0 and self._arch_hw_profile_active:
                Logger.warning(
                    "Architecture-derived CIMLoop hardware profile failed during generation; "
                    "retrying with stable default hardware profile"
                )
                self._activate_default_hw_profile()
                docker_rc, docker_out = self._run_cimloop_generate_layer_docker(
                    layer_arg=docker_layer_arg,
                    workload_obj=workload_obj,
                )

            if docker_rc != 0 and self._is_mapper_infeasible_output(docker_out):
                for profile in self._compatibility_hw_profiles:
                    Logger.warning(
                        "CIMLoop mapper infeasible for current profile; "
                        f"retrying with compatibility profile {profile['name']} "
                        f"(macro={profile['macro']}, system={profile['system']})"
                    )
                    self._apply_hw_profile(profile, arch_active=False)
                    docker_rc, docker_out = self._run_cimloop_generate_layer_docker(
                        layer_arg=docker_layer_arg,
                        workload_obj=workload_obj,
                    )
                    if docker_rc == 0:
                        break

            self._restore_hw_profile(original_hw_profile)
            if docker_rc != 0:
                local_tail = "\n".join(local_out.splitlines()[-80:])
                docker_tail = "\n".join(docker_out.splitlines()[-80:])
                raise ValueError(
                    "CIMLoop generation failed in both host and docker environments. "
                    f"Host tail:\n{local_tail}\n\nDocker tail:\n{docker_tail}"
                )

            docker_payload = self._extract_docker_mapping_payload(docker_out)
            if docker_payload is None:
                docker_tail = "\n".join(docker_out.splitlines()[-120:])
                raise ValueError(
                    "Docker CIMLoop run finished but mapping payload was not captured from stdout. "
                    f"Runner tail:\n{docker_tail}"
                )

            result_json.write_text(json.dumps(docker_payload), encoding="utf-8")
        elif local_rc != 0:
            tail = "\n".join(local_out.splitlines()[-120:])
            raise ValueError(f"CIMLoop run failed (exit={local_rc}).\n{tail}")

        if not result_json.is_file():
            raise ValueError(f"CIMLoop run finished but result marker not found: {result_json}")

        payload = json.loads(result_json.read_text(encoding="utf-8"))
        mapping_text = payload.get("mapping")
        if not isinstance(mapping_text, str) or len(mapping_text.strip()) == 0:
            raise ValueError(
                "CIMLoop run finished but no mapping text was returned. "
                f"Please inspect runner output directory: {layer_output_dir}"
            )

        sections = self._split_loop_tokens(mapping_text)
        if len(sections) == 0:
            raise ValueError("Failed to parse mapping text emitted by CIMLoop")

        baseline_payload = self._build_baseline_yaml_payload(loop_dim=loop_dim, sections=sections)
        baseline_path = layer_output_dir / "baseline_layer.yaml"
        with open(baseline_path, "w", encoding="utf-8") as fp:
            yaml.safe_dump(baseline_payload, fp, sort_keys=False)
        (layer_output_dir / "timeloop-mapper.map.txt").write_text(mapping_text, encoding="utf-8")
        Logger.info(f"Generated CIMLoop baseline layer file: {baseline_path}")

    def find_layer(self, loop_dim: dict[str, int]) -> BaselineLayer:
        key = _loop_key(loop_dim)
        cached = self._layers_by_key.get(key)
        if cached is not None:
            return cached

        if self._mode == "map":
            raise ValueError(
                f"No CIMLoop baseline layer found for loop_dim={loop_dim}. "
                f"Loaded keys={list(self._layers_by_key.keys())}"
            )

        label = _loop_label(loop_dim)
        prob_path = self._input_workload_dir / f"{label}.yaml"
        layer_output_dir = self._output_root / label
        self._export_prob_yaml(loop_dim, prob_path)
        Logger.info(f"Generated CIMLoop workload for {label}: {prob_path}")

        generated_files = self._discover_generated_layer_files(layer_output_dir)
        if len(generated_files) == 0:
            self._run_cimloop_generate_layer(loop_dim=loop_dim, prob_path=prob_path, layer_output_dir=layer_output_dir)
            generated_files = self._discover_generated_layer_files(layer_output_dir)

        if len(generated_files) == 0:
            raise ValueError(
                "CIMLoop generate mode exported workload but no baseline mapping was found. "
                f"Please run CIMLoop to generate layer baseline YAML under {layer_output_dir} and retry. "
                f"Exported workload: {prob_path}"
            )

        layer = parse_cimloop_baseline_yaml(generated_files[0])
        self._layers_by_key[key] = layer
        Logger.info(f"Loaded generated CIMLoop baseline for {label}: {generated_files[0]}")
        return layer
