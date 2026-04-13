from pathlib import Path
import importlib
import os
import shutil
import subprocess
import sys


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cosa_submodule_root() -> Path:
    return repo_root() / "Evaluation" / "CoSA" / "cosa"


def cosa_package_root() -> Path:
    return cosa_submodule_root() / "src"


def cosa_entry_file() -> Path:
    return cosa_package_root() / "cosa" / "cosa.py"


def cosa_default_mapspace_path() -> Path:
    return cosa_package_root() / "cosa" / "configs" / "mapspace" / "mapspace.yaml"


def _infer_timeloop_root() -> Path | None:
    env_root = os.environ.get("TIMELOOP_DIR")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if candidate.is_dir():
            return candidate

    repo_candidate = repo_root() / "timeloop"
    if repo_candidate.is_dir():
        return repo_candidate.resolve()

    for binary_name in ("timeloop-model", "timeloop-mapper"):
        binary = shutil.which(binary_name)
        if binary is None:
            continue
        binary_path = Path(binary).expanduser().resolve()
        if binary_path.parent.name == "bin" and binary_path.parent.parent.is_dir():
            return binary_path.parent.parent
    return None


def _missing_python_modules(module_names: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for name in module_names:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001
            missing.append(name)
    return missing


def ensure_cosa_submodule_on_path() -> Path:
    api_file = cosa_entry_file()
    if not api_file.is_file():
        raise FileNotFoundError(f"CoSA submodule is missing: {api_file}")

    package_root = str(cosa_package_root())
    if package_root in sys.path:
        sys.path.remove(package_root)
    sys.path.insert(0, package_root)
    return cosa_submodule_root()


def assert_cosa_local_ready() -> Path:
    ensure_cosa_submodule_on_path()

    missing_modules = _missing_python_modules(("gurobipy", "numpy", "yaml"))
    missing_bins = [name for name in ("timeloop-model", "timeloop-mapper") if shutil.which(name) is None]
    timeloop_root = _infer_timeloop_root()

    problems: list[str] = []
    if missing_modules:
        problems.append(f"missing Python modules: {', '.join(missing_modules)}")
    if missing_bins:
        problems.append(f"missing Timeloop CLI tools: {', '.join(missing_bins)}")
    if timeloop_root is None:
        problems.append("unable to infer TIMELOOP_DIR from env, repo-local timeloop/, or Timeloop CLI path")

    if problems:
        detail = "\n".join(f"- {item}" for item in problems)
        raise RuntimeError(
            "CoSA local generate mode requires a preinstalled local environment.\n"
            f"{detail}\n"
            "Use `--cosa-map` to replay an existing CoSA mapping, or install the local CoSA/Timeloop runtime first."
        )
    return timeloop_root


def cosa_runtime_env() -> dict[str, str]:
    timeloop_root = assert_cosa_local_ready()
    env = os.environ.copy()
    package_root = str(cosa_package_root())
    env["PYTHONPATH"] = (
        f"{package_root}:{env.get('PYTHONPATH', '')}"
        if env.get("PYTHONPATH")
        else package_root
    )
    env["TIMELOOP_DIR"] = str(timeloop_root)
    return env


def run_cosa(prob_path: Path, arch_path: Path, mapspace_path: Path, output_dir: Path):
    cmd = [
        sys.executable,
        "-m",
        "cosa.cosa",
        "-o",
        str(output_dir),
        "-ap",
        str(arch_path),
        "-mp",
        str(mapspace_path),
        "-pp",
        str(prob_path),
    ]
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        env=cosa_runtime_env(),
    )
