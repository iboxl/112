from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import re

from Architecture.ArchSpec import CIM_Acc


@dataclass
class CimloopHardwareSpec:
    macro: str
    system: str
    tile: str | None
    chip: str | None
    iso: str | None
    variables: dict[str, int | float | bool]
    source: str


class CimloopHardwareBridge:
    """Build a CIMLoop hardware profile from MIREDO Architecture (CIM_Acc)."""

    def __init__(self, architecture: str, cimloop_root: Path, base_macro: str = "isaac_isca_2016"):
        self.architecture = architecture
        self.cimloop_root = Path(cimloop_root)
        self.base_macro = base_macro

    @staticmethod
    def _replace_scalar(text: str, key: str, value: int) -> str:
        pattern = rf"(^\s*{re.escape(key)}:\s*).*$"
        return re.sub(pattern, rf"\g<1>{value}", text, flags=re.MULTILINE)

    @staticmethod
    def _replace_mesh_after_name(text: str, name: str, axis: str, value: int) -> str:
        pattern = rf"(name:\s*{re.escape(name)}[\s\S]*?spatial:\s*\{{{axis}:\s*)\d+"
        return re.sub(pattern, rf"\g<1>{value}", text, count=1)

    @staticmethod
    def _replace_width_after_name(text: str, name: str, value: int) -> str:
        pattern = rf"(name:\s*{re.escape(name)}[\s\S]*?width:\s*)\d+"
        return re.sub(pattern, rf"\g<1>{value}", text, count=1)

    @staticmethod
    def _find_mem_index(acc: CIM_Acc, name: str) -> int:
        for m in range(1, acc.Num_mem):
            if acc.mem2dict(m) == name:
                return m
        raise ValueError(f"Memory '{name}' not found in accelerator")

    def _build_acc(self) -> CIM_Acc:
        acc_template = import_module(f"Architecture.{self.architecture}").accelerator
        return CIM_Acc(acc_template.cores[0])

    def _emit_generated_macro(self, acc: CIM_Acc) -> tuple[str, str]:
        models_root = self.cimloop_root / "workspace" / "models"
        macro_root = models_root / "arch" / "1_macro"
        base_dir = macro_root / self.base_macro
        if not base_dir.is_dir():
            raise FileNotFoundError(f"Base CIMLoop macro not found: {base_dir}")

        macro_name = f"__miredo_{self.architecture.lower()}"
        gen_dir = macro_root / macro_name
        gen_dir.mkdir(parents=True, exist_ok=True)

        input_idx = self._find_mem_index(acc, "Input_buffer")
        output_idx = self._find_mem_index(acc, "Output_buffer")
        macro_idx = self._find_mem_index(acc, "Macro")

        input_bits = int(acc.precision[input_idx, 0])
        weight_bits = int(acc.precision[macro_idx, 1])
        output_bits = int(acc.precision[output_idx, 2])
        input_bw = max(1, int(acc.bw[input_idx]))
        output_bw = max(1, int(acc.bw[output_idx]))
        n_cores = max(1, int(acc.Num_core))
        depth_cells = max(1, int(acc.memSize[macro_idx]) // max(1, weight_bits))

        arch_text = (base_dir / "arch.yaml").read_text(encoding="utf-8")
        arch_text = self._replace_mesh_after_name(arch_text, "array", "meshX", n_cores)
        # Keep row/column fanout from the base macro template.
        # Directly rewriting these two values from Architecture dimX/dimY often
        # leads to no-valid-mapping failures in timeloop mapper for CIMLoop.
        arch_text = self._replace_width_after_name(arch_text, "input_buffer", input_bw)
        arch_text = self._replace_width_after_name(arch_text, "output_buffer", output_bw)
        (gen_dir / "arch.yaml").write_text(arch_text, encoding="utf-8")

        iso_text = (base_dir / "variables_iso.yaml").read_text(encoding="utf-8")
        iso_text = self._replace_scalar(iso_text, "INPUT_BITS", input_bits)
        iso_text = self._replace_scalar(iso_text, "WEIGHT_BITS", weight_bits)
        iso_text = self._replace_scalar(iso_text, "OUTPUT_BITS", output_bits)
        iso_text = self._replace_scalar(iso_text, "SUPPORTED_INPUT_BITS", input_bits)
        iso_text = self._replace_scalar(iso_text, "SUPPORTED_WEIGHT_BITS", weight_bits)
        iso_text = self._replace_scalar(iso_text, "SUPPORTED_OUTPUT_BITS", output_bits)
        (gen_dir / "variables_iso.yaml").write_text(iso_text, encoding="utf-8")

        free_text = (base_dir / "variables_free.yaml").read_text(encoding="utf-8")
        free_text = self._replace_scalar(free_text, "CIM_UNIT_DEPTH_CELLS", depth_cells)
        (gen_dir / "variables_free.yaml").write_text(free_text, encoding="utf-8")

        return macro_name, macro_name

    def build(self) -> CimloopHardwareSpec:
        acc = self._build_acc()

        macro_name, iso_name = self._emit_generated_macro(acc)
        system = "ws_dummy_buffer_many_macro" if int(acc.Num_core) > 1 else "ws_dummy_buffer_one_macro"

        input_idx = self._find_mem_index(acc, "Input_buffer")
        output_idx = self._find_mem_index(acc, "Output_buffer")
        macro_idx = self._find_mem_index(acc, "Macro")

        variables: dict[str, int | float | bool] = {
            "CIM_ARCHITECTURE": True,
            "INPUT_BITS": int(acc.precision[input_idx, 0]),
            "WEIGHT_BITS": int(acc.precision[macro_idx, 1]),
            "OUTPUT_BITS": int(acc.precision[output_idx, 2]),
            "SUPPORTED_INPUT_BITS": int(acc.precision[input_idx, 0]),
            "SUPPORTED_WEIGHT_BITS": int(acc.precision[macro_idx, 1]),
            "SUPPORTED_OUTPUT_BITS": int(acc.precision[output_idx, 2]),
            "BATCH_SIZE": 1,
            "N_ADC_PER_BANK": 1,
            "CIM_UNIT_DEPTH_CELLS": max(1, int(acc.memSize[macro_idx]) // max(1, int(acc.precision[macro_idx, 1]))),
        }

        return CimloopHardwareSpec(
            macro=macro_name,
            system=system,
            tile=None,
            chip=None,
            iso=iso_name,
            variables=variables,
            source=f"Architecture.{self.architecture}",
        )
