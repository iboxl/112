from pathlib import Path
import os
import shutil
import sys


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cimloop_submodule_root() -> Path:
    return repo_root() / "Evaluation" / "CIMLoop" / "cimloop"


def cimloop_workspace_root() -> Path:
    return cimloop_submodule_root() / "workspace"


def cimloop_models_root() -> Path:
    return cimloop_workspace_root() / "models"


def cimloop_scripts_root() -> Path:
    return cimloop_workspace_root() / "scripts"


def cimloop_infra_root() -> Path:
    return repo_root() / "Evaluation" / "CIMLoop" / "timeloop-accelergy-infra"


def cimloop_timeloop_root() -> Path:
    return cimloop_infra_root() / "src" / "timeloop"


def cimloop_timeloop_bin_root() -> Path:
    return cimloop_timeloop_root() / "bin"


def cimloop_timeloop_lib_root() -> Path:
    return cimloop_timeloop_root() / "build"


def cimloop_output_root() -> Path:
    output_dir = repo_root() / "Evaluation" / "CIMLoop" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def cimloop_cache_path(macro: str, model: str, layer_tag: str, objective: str) -> Path:
    return cimloop_output_root() / f"{objective}_{model}_{macro}_{layer_tag}.pickle"


def is_cimloop_available() -> bool:
    if shutil.which("timeloop-mapper") is None:
        return False
    # 尝试 shim 后检查
    ensure_pytimeloop_shim()
    try:
        import pytimeloop.timeloopfe.v4  # noqa: F401
        return True
    except (ImportError, AttributeError):
        pass
    # 回退: 独立 timeloopfe 也行
    try:
        import timeloopfe.v4  # noqa: F401
        return True
    except ImportError:
        return False


def ensure_cimloop_available():
    missing = []
    if shutil.which("timeloop-mapper") is None:
        missing.append("timeloop-mapper binary (not on PATH)")

    ensure_pytimeloop_shim()
    has_tl_api = False
    try:
        import pytimeloop.timeloopfe.v4  # noqa: F401
        has_tl_api = True
    except (ImportError, AttributeError):
        pass
    if not has_tl_api:
        try:
            import timeloopfe.v4  # noqa: F401
            has_tl_api = True
        except ImportError:
            pass
    if not has_tl_api:
        missing.append("timeloopfe (Python package)")

    if not missing:
        return

    raise RuntimeError(
        "CIMLoop dependencies incomplete. Missing:\n"
        + "".join(f"  - {m}\n" for m in missing)
        + "\nRun:  bash Evaluation/CIMLoop/setup_cimloop.sh\n"
        + "Or install Python packages only:\n"
        + "  cd Evaluation/CIMLoop/timeloop-accelergy-infra\n"
        + "  git submodule update --init src/timeloopfe\n"
        + "  pip install ./src/timeloopfe"
    )


def ensure_cimloop_submodule():
    marker = cimloop_models_root() / "top.yaml.jinja2"
    if not marker.is_file():
        raise FileNotFoundError(
            f"CIMLoop submodule is missing: {marker}\n"
            "Run: git submodule update --init --recursive"
        )


def ensure_pytimeloop_shim():
    """让 import pytimeloop.timeloopfe.v4 能工作。

    pytimeloop 的 pip install 需要构建 C++ 扩展 (pybind11 bindings),
    但 CIMLoop 实际只用 pytimeloop.timeloopfe.v4 (纯 Python, 通过
    subprocess 调 timeloop-mapper)。timeloopfe 可独立安装。

    此 shim 将已安装的 timeloopfe 包注册为 pytimeloop.timeloopfe,
    无需 C++ 编译。
    """
    if "pytimeloop" in sys.modules:
        return
    try:
        import pytimeloop  # noqa: F401
        return  # 已正式安装, 无需 shim
    except ImportError:
        pass

    import types
    try:
        import timeloopfe
    except ImportError:
        return  # timeloopfe 也没有, ensure_cimloop_available() 会报错

    pytimeloop_mod = types.ModuleType("pytimeloop")
    pytimeloop_mod.__path__ = []
    pytimeloop_mod.timeloopfe = timeloopfe
    sys.modules["pytimeloop"] = pytimeloop_mod
    sys.modules["pytimeloop.timeloopfe"] = timeloopfe
    # 注册 v4 子模块
    if hasattr(timeloopfe, "v4"):
        sys.modules["pytimeloop.timeloopfe.v4"] = timeloopfe.v4


def ensure_cimloop_scripts_on_path():
    ensure_pytimeloop_shim()
    scripts_dir = str(cimloop_scripts_root())
    while scripts_dir in sys.path:
        sys.path.remove(scripts_dir)
    sys.path.append(scripts_dir)


def cimloop_runtime_env() -> dict[str, str]:
    """Return a shell-safe Timeloop runtime env for timeloopfe.call_mapper().

    timeloopfe prefixes these entries as raw ``KEY=VALUE`` shell tokens, so do
    not forward the full process environment. Values such as
    ``CONDA_PROMPT_MODIFIER=(MIREDO)`` break the generated shell command, and a
    polluted LD_LIBRARY_PATH can make CiMLoop load another baseline's Timeloop.
    """
    bin_dir = str(cimloop_timeloop_bin_root())
    lib_dir = str(cimloop_timeloop_lib_root())
    conda_lib = str(Path(sys.prefix) / "lib")
    env = {
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "LD_LIBRARY_PATH": ":".join(part for part in [lib_dir, conda_lib] if part),
        "TIMELOOP_OUTPUT_STAT_SCIENTIFIC": "1",
        "TIMELOOP_OUTPUT_STAT_DEFAULT_FLOAT": "0",
        "TIMELOOP_HIDE_INCONSEQUENTIAL_STATS": "0",
    }
    return {key: value for key, value in env.items() if value}
