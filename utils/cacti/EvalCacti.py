import subprocess, tempfile, re, os, pathlib

def _patched_cfg_text(template_path: str,
                      cap_bytes: int,
                      bitwidth_bits: int) -> str:
    targets = {
        "-size":       cap_bytes,
        "-block size": bitwidth_bits // 8,
        "-bus width":  bitwidth_bits,
    }
    out_lines = []
    with open(template_path) as fh:
        for ln in fh:
            low = ln.lstrip().lower()
            replaced = False
            for key, val in targets.items():
                if low.startswith(key):
                    out_lines.append(f"{key} {val}\n")
                    replaced = True
                    break
            if not replaced:
                out_lines.append(ln)
    # 若模板缺某字段，追加
    for k, v in targets.items():
        if not any(ln.lstrip().lower().startswith(k) for ln in out_lines):
            out_lines.append(f"{k} {v}\n")
    return "".join(out_lines)

def cacti_power(capacity_bytes: int,
                bitwidth_bits: int,
                *,
                cacti_bin: str = "utils/cacti/fncacti/cacti",
                template_cfg: str = "utils/cacti/fncacti/configs/config.cfg",
                col: int = 4):
    cfg_txt = _patched_cfg_text(template_cfg, capacity_bytes, bitwidth_bits)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".cfg", mode="w") as tmp:
        tmp.write(cfg_txt)
        tmp.flush()
        cfg_path = tmp.name

    try:
        stdout = subprocess.check_output(
            [str(pathlib.Path(cacti_bin).expanduser()), "-infile", cfg_path],
            text=True, stderr=subprocess.STDOUT
        )
    finally:
        os.unlink(cfg_path)           # 立刻删除

    ser = r"([0-9.+\-eE]+(?:\s*;\s*[0-9.+\-eE]+)*)"
    def pick(pattern: str) -> float:
        m = re.search(pattern + ser, stdout)
        if not m:
            raise RuntimeError("Regex failed:\n" + stdout[:200])
        return float(m.group(1).split(";")[col].strip())

    read_nJ  = pick(r"Total dynamic read energy.*?\(nJ\):\s*")
    write_nJ = pick(r"Total dynamic write energy.*?\(nJ\):\s*")
    leak_mW  = pick(r"Total leakage power.*?\(mW\):\s*")
    cycle_ns = pick(r"Cycle time.*?\(ns\):\s*")

    leak_pJ_per_cycle = leak_mW * cycle_ns     # mW·ns → pJ
    # print(f"cycle_ns: {cycle_ns}")
    return read_nJ * 1e3, write_nJ*1e3, leak_pJ_per_cycle , leak_mW     # Unit pJ

def dram_static(
    capacity_bytes: int,
    bus_width_bits: int = 64,
    mem_type: str = "DDR4",  # {"DDR4", "LPDDR4", "HBM2"}
    freq_hz: float = 500e6,  # 500 MHz # 若给定，则同时返回 pJ/周期
):
    """
    Parameters
    ----------
    capacity_bytes : int  DRAM 容量 (B)
    bus_width_bits : int  通讯总线宽 (bit)；典型 DDR/LPDDR 为 16、32、64
    mem_type       : str  DRAM 类型
    freq_hz        : float | None  时钟频率Hz若给返回 (P_leak_mW, E_leak_pJ)

    Returns
    -------
    P_leak_mW               : float  mW
    (可选) E_leak_per_cycle : float  pJ / cycle
    """
    # —— 每 “GiB·64-bit” 通道的典型漏功耗 (mW) ——  资料见下表
    LEAK_PER_GB = {
        "DDR4":   42.0,   # 8 Gb x16 DDR4: IDD2N=35 mA @1.2 V  → 35*1.2 ≈ 42 mW :contentReference[oaicite:0]{index=0}
        "LPDDR4":  9.0,   # 8 Gb LPDDR4: IDD2N≈8 mA @1.1 V    → 8*1.1  ≈ 8.8 mW :contentReference[oaicite:1]{index=1}
        "HBM2":   20.0,   # 行业白皮书公开均值（Samsung/AMD）
    }
    if mem_type not in LEAK_PER_GB:
        raise ValueError(f"mem_type must be one of {list(LEAK_PER_GB)}")

    cap_gib  = capacity_bytes / 2**30  # GiB
    width_fac = bus_width_bits / 64    # 相对 64-bit 通道的比例

    P_leak_mW = LEAK_PER_GB[mem_type] * cap_gib * width_fac

    if freq_hz is None:
        return P_leak_mW
    # 能量 = 功率 × 周期
    E_leak_pJ = P_leak_mW * 1e-3 / freq_hz * 1e12
    return P_leak_mW, E_leak_pJ


if __name__ == "__main__":
    er, ew, el, pl = cacti_power(
        capacity_bytes = 32 * 1024,        # 32 KiB
        bitwidth_bits  = 128,              # 128-bit
        cacti_bin      = "utils/cacti/fncacti/cacti",
        template_cfg   = "utils/cacti/fncacti/configs/config.cfg",
        col = 0                             # 这里选“最低泄漏功耗”列
    )
    print(f"Read  : {er:.3f} pJ  per access")
    print(f"Write : {ew:.3f} pJ  per access")
    print(f"LeakE : {el:.3f}  pJ  per cycle")
    print(f"Leak  : {pl:.3f}  mW  (static power)")




