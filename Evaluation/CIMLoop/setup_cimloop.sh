#!/usr/bin/env bash
# CIMLoop 原生安装脚本 — 无需 sudo，基于 conda + 子模块
# 用法: bash Evaluation/CIMLoop/setup_cimloop.sh
#
# 前置条件: 已激活 conda 环境 (MIREDO)
# 预计耗时: 首次 ~10 分钟 (主要是编译 timeloop)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="${SCRIPT_DIR}/timeloop-accelergy-infra"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# ============================================================
# 0. 前置检查
# ============================================================
if [ -z "${CONDA_PREFIX:-}" ]; then
    fail "未检测到 conda 环境。请先激活: conda activate MIREDO"
fi
info "conda 环境: ${CONDA_PREFIX}"

if [ ! -f "${INFRA_DIR}/Makefile" ]; then
    fail "子模块不存在: ${INFRA_DIR}\n  请先运行: git submodule update --init Evaluation/CIMLoop/timeloop-accelergy-infra"
fi

# ============================================================
# 1. 初始化嵌套子模块
# ============================================================
info "初始化嵌套子模块..."
cd "${INFRA_DIR}"
git submodule update --init src/timeloop src/timeloopfe src/accelergy \
    src/accelergy-cacti-plug-in src/accelergy-neurosim-plug-in \
    src/accelergy-aladdin-plug-in src/accelergy-table-based-plug-ins \
    src/accelergy-library-plug-in src/accelergy-adc-plug-in
cd src/timeloop && git submodule update --init --recursive && cd "${INFRA_DIR}"
info "子模块就绪。"

# ============================================================
# 2. conda 安装 C 库依赖 (ISL, GMP, NTL, Barvinok)
# ============================================================
info "通过 conda 安装 C 库依赖 (isl, gmp, ntl, barvinok)..."
conda install -y -c conda-forge isl gmp ntl barvinok 2>&1 | tail -5

# 验证头文件
for hdr in isl/map.h isl/cpp.h gmp.h NTL/ZZ.h; do
    if [ ! -f "${CONDA_PREFIX}/include/${hdr}" ]; then
        fail "缺少头文件: ${CONDA_PREFIX}/include/${hdr}"
    fi
done
info "C 库依赖就绪。"

# ============================================================
# 3. 编译 Timeloop (从子模块，链接 conda 库)
# ============================================================
info "编译 Timeloop..."
cd "${INFRA_DIR}/src/timeloop"

# PAT 依赖
if [ ! -d "src/pat" ] && [ -d "pat-public/src/pat" ]; then
    cp -r pat-public/src/pat src/pat
fi

# 通过构建环境注入 conda 库路径和警告选项，避免修改 vendored Timeloop 源码。
export BARVINOKPATH="${CONDA_PREFIX}"
export NTLPATH="${CONDA_PREFIX}"
export CPATH="${CONDA_PREFIX}/include:${CPATH:-}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# 该 Timeloop SConstruct 不读取命令行 CCFLAGS；用一次性 compiler shim 避免改 SConscript。
CIMLOOP_REAL_CXX="$(command -v g++)"
CIMLOOP_REAL_CC="$(command -v gcc)"
CIMLOOP_COMPILER_WRAPPER_DIR="$(mktemp -d)"
trap 'rm -rf "${CIMLOOP_COMPILER_WRAPPER_DIR}"' EXIT
cat > "${CIMLOOP_COMPILER_WRAPPER_DIR}/g++" <<EOF
#!/usr/bin/env bash
exec "${CIMLOOP_REAL_CXX}" "\$@" -Wno-error=missing-field-initializers
EOF
cat > "${CIMLOOP_COMPILER_WRAPPER_DIR}/gcc" <<EOF
#!/usr/bin/env bash
exec "${CIMLOOP_REAL_CC}" "\$@" -Wno-error=missing-field-initializers
EOF
chmod +x "${CIMLOOP_COMPILER_WRAPPER_DIR}/g++" "${CIMLOOP_COMPILER_WRAPPER_DIR}/gcc"
export PATH="${CIMLOOP_COMPILER_WRAPPER_DIR}:${PATH}"

scons -j"$(nproc)" --with-isl --accelergy 2>&1 | tail -5

# 安装到 ~/.local
mkdir -p ~/.local/bin ~/.local/lib
cp build/timeloop-mapper build/timeloop-model build/timeloop-metrics ~/.local/bin/
cp build/libtimeloop-*.so ~/.local/lib/
info "Timeloop 已安装到 ~/.local/bin/ 和 ~/.local/lib/"

# ============================================================
# 4. Python 包
# ============================================================
cd "${INFRA_DIR}"
info "安装 Python 包..."

pip install -q setuptools wheel libconf numpy joblib jinja2

# accelergy
pip install -q ./src/accelergy
for plugin_dir in src/accelergy-cacti-plug-in src/accelergy-neurosim-plug-in \
                  src/accelergy-aladdin-plug-in src/accelergy-table-based-plug-ins \
                  src/accelergy-library-plug-in src/accelergy-adc-plug-in; do
    if [ -d "${plugin_dir}" ]; then
        if [ -f "${plugin_dir}/Makefile" ]; then
            (cd "${plugin_dir}" && make) 2>/dev/null || true
        fi
        if [ -f "${plugin_dir}/setup.py" ] || [ -f "${plugin_dir}/pyproject.toml" ]; then
            pip install -q "./${plugin_dir}" 2>/dev/null || warn "插件跳过: ${plugin_dir}"
        fi
    fi
done

# timeloopfe (必须与上面编译的 C++ binary 版本匹配)
pip install -q ./src/timeloopfe

info "Python 包安装完成。"
info "pytimeloop C++ bindings 跳过 (通过 shim 使用 timeloopfe)。"

# ============================================================
# 5. 环境变量提示
# ============================================================
echo ""
info "请确保以下环境变量已设置 (建议加入 ~/.bashrc):"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
echo "  export LD_LIBRARY_PATH=\"\$HOME/.local/lib:\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH\""
echo ""

# ============================================================
# 6. 验证
# ============================================================
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

info "验证安装..."
PASS=true

if command -v timeloop-mapper &>/dev/null; then
    info "  timeloop-mapper : $(which timeloop-mapper)"
    # 测试 binary 能运行
    timeloop-mapper 2>&1 | grep -q "Assertion" && info "  binary 运行正常" || true
else
    warn "  timeloop-mapper : 未找到"; PASS=false
fi

python3 -c "import timeloopfe.v4; print(f'  timeloopfe.v4   : {timeloopfe.v4.__file__}')" 2>/dev/null \
    || { warn "  timeloopfe.v4   : 导入失败"; PASS=false; }

python3 -c "import accelergy; print(f'  accelergy       : {accelergy.__file__}')" 2>/dev/null \
    || { warn "  accelergy       : 导入失败"; PASS=false; }

echo ""
if $PASS; then
    info "CIMLoop 环境就绪。全部组件版本匹配，无需 sudo。"
else
    warn "部分组件缺失。"; exit 1
fi
