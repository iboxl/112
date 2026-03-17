from pathlib import Path
import sys

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def zigzag_submodule_root() -> Path:
    return repo_root() / "Evaluation" / "Zigzag_imc" / "zigzag-imc"


def zigzag_package_root() -> Path:
    return zigzag_submodule_root() / "zigzag"


def zigzag_cacti_root() -> Path:
    return zigzag_package_root() / "classes" / "cacti" / "cacti_master"


def ensure_zigzag_submodule_on_path() -> Path:
    root = zigzag_submodule_root()
    api_file = root / "zigzag" / "api.py"
    if not api_file.is_file():
        raise FileNotFoundError(f"ZigZag submodule is missing: {api_file}")

    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    return root


def zigzag_output_root() -> Path:
    output_dir = repo_root() / "Evaluation" / "Zigzag_imc" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def zigzag_cache_prefix(opt_flag: str, model: str, architecture: str) -> Path:
    return zigzag_output_root() / f"{opt_flag}_{model}_{architecture}"


def get_hardware_performance_zigzag(*args, **kwargs):
    ensure_zigzag_submodule_on_path()
    from zigzag.api import get_hardware_performance_zigzag as zigzag_api
    return zigzag_api(*args, **kwargs)


def convert_Zigzag_to_MIREDO(*args, **kwargs):
    from Evaluation.Zigzag_imc.CompatibleZigzag import convert_Zigzag_to_MIREDO as compat_api
    return compat_api(*args, **kwargs)


def compare_ops_cme(*args, **kwargs):
    from Evaluation.Zigzag_imc.CompatibleZigzag import compare_ops_cme as compat_api
    return compat_api(*args, **kwargs)
