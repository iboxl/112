#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path-to-112.log>"
  exit 2
fi

log_path="$1"
if [[ ! -f "$log_path" ]]; then
  echo "[ERROR] Log file not found: $log_path"
  exit 2
fi

count_pattern() {
  local pattern="$1"
  rg -c "$pattern" "$log_path" 2>/dev/null || true
}

memsize_count="$(count_pattern "Dataflow Over MemSize Error")"
gbound_count="$(count_pattern "Illegal dimension G bound in LoopNest")"
unroll_count="$(count_pattern "incomplete unrolling before replay")"
ilevel_count="$(count_pattern "operand 'I' has")"
unavail_count="$(count_pattern "Optimization summary unavailable")"

echo "[CHECK] $log_path"
echo "- Dataflow Over MemSize Error        : ${memsize_count:-0}"
echo "- Illegal dimension G bound in LoopNest: ${gbound_count:-0}"
echo "- incomplete unrolling before replay : ${unroll_count:-0}"
echo "- operand 'I' has                    : ${ilevel_count:-0}"
echo "- Optimization summary unavailable   : ${unavail_count:-0}"

if [[ "${memsize_count:-0}" -eq 0 \
   && "${gbound_count:-0}" -eq 0 \
   && "${unroll_count:-0}" -eq 0 \
   && "${ilevel_count:-0}" -eq 0 \
   && "${unavail_count:-0}" -eq 0 ]]; then
  echo "[PASS] All critical regression checks are zero."
  exit 0
fi

echo "[FAIL] At least one critical regression check is non-zero."
exit 1