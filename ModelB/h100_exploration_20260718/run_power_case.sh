#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mode=${1:?usage: run_power_case.sh MODE [WINDOW_SECONDS] [SOLVE_T_END]}
window_seconds=${2:-20}
solve_t_end=${3:-2.0}
stamp=$(date +%Y%m%d_%H%M%S)
result_dir="$script_dir/power_results/${stamp}_${mode}"
mkdir -p "$result_dir"

dcgm_log="$result_dir/dcgm.log"
smi_log="$result_dir/nvidia_smi.csv"
workload_log="$result_dir/workload.log"
metadata_log="$result_dir/metadata.txt"

{
    hostname
    uname -m
    sha256sum "$script_dir/../ModelB.jl"
    nvidia-smi --query-gpu=name,uuid,pci.bus_id,power.limit,power.default_limit,power.max_limit,clocks.max.sm,clocks.max.memory --format=csv,noheader,nounits
    env -u CUDA_VISIBLE_DEVICES dcgmi dmon -e 155,157,164,100,101 -i 0 -c 1
} >"$metadata_log" 2>&1

cleanup() {
    if [[ -n "${dcgm_pid:-}" ]]; then kill "$dcgm_pid" 2>/dev/null || true; fi
    if [[ -n "${smi_pid:-}" ]]; then kill "$smi_pid" 2>/dev/null || true; fi
    wait "${dcgm_pid:-}" 2>/dev/null || true
    wait "${smi_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# DCGM's profiling fields are sampled at 100 ms.  Prefix each line as it is
# received so the parser can restrict statistics to the explicit Julia marker
# window instead of including compilation and allocation.
env -u CUDA_VISIBLE_DEVICES stdbuf -oL dcgmi dmon \
    -e 155,156,157,164,240,100,101,1001,1002,1003,1004,1005,1007,1016 \
    -i 0 -d 100 \
    | perl -MTime::HiRes=time -ne 'printf "%.6f %s", time(), $_' \
    >"$dcgm_log" 2>&1 &
dcgm_pid=$!

nvidia-smi \
    --query-gpu=timestamp,power.draw,clocks.current.graphics,clocks.current.memory,utilization.gpu,utilization.memory,temperature.gpu,pstate,clocks_event_reasons.active \
    --format=csv,noheader,nounits --loop-ms=100 >"$smi_log" 2>&1 &
smi_pid=$!

sleep 0.5
if [[ "$mode" == "gemm_control" ]]; then
    julia "$script_dir/power_control.jl" 16384 "$window_seconds" 2>&1 | tee "$workload_log"
elif [[ "$mode" == "fft_drift_only" ]]; then
    julia "$script_dir/power_fft_workload.jl" "$window_seconds" 32 2>&1 | tee "$workload_log"
elif [[ "$mode" == "custom_fused_philox" || "$mode" == "fused_fixed_sosra" ]]; then
    julia "$script_dir/power_fused_workload.jl" "$window_seconds" "$solve_t_end" 1024 264 512 528 2>&1 | tee "$workload_log"
else
    julia "$script_dir/power_workload.jl" "$mode" "$window_seconds" "$solve_t_end" 2>&1 | tee "$workload_log"
fi
sleep 0.5
cleanup
trap - EXIT INT TERM

python3 "$script_dir/summarize_power.py" "$result_dir"
echo "$result_dir"
