#!/bin/bash -l

set -euo pipefail

cd /work/gw97/w97000/project/DM/ModelB
source /etc/profile.d/modules.sh
module load miniforge3
source activate nenv2

OUT=$PWD/runs/multiL_ep10000_reverse_analysis
mkdir -p "$OUT/reference"

echo "Node: $(hostname)"
echo "Start: $(date '+%F %T')"
nvidia-smi -L

reference="$OUT/reference/modelB_L64_Ne512_hmc_t0p5_dt0.002_seed2026080701.jld2"
if [[ ! -e "$reference" ]]; then
    julia --startup-file=no -e '
using CUDA, JLD2, Statistics
include("solver/ModelB_fixed.jl")
source = "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip/samples/hmc_L64_N512_seed2026080201/canonical_L64_N512_k0p340340_l1_seed2026080201.jld2"
output = "runs/multiL_ep10000_reverse_analysis/reference/modelB_L64_Ne512_hmc_t0p5_dt0.002_seed2026080701.jld2"
@assert !ispath(output)
cfgs = load(source, "cfgs")
M0 = vec(mean(Float64, cfgs; dims=(1, 2)))
u = CuArray(cfgs)
seed = UInt64(2026080701)
seconds = @elapsed begin
    buffers = modelb_fixed_buffers(u; seed)
    modelB_2d_fixed!(u, buffers; γ=1.0f0, T=1.0f0, κ=0.340340f0,
                     λ=1.0f0, dt=0.002f0, tspan=(0.0f0, 0.5f0))
    CUDA.synchronize()
end
cfgs = Array(u)
drift = maximum(abs, vec(mean(Float64, cfgs; dims=(1, 2))) - M0)
jldsave(output; cfgs, L=64, Nensemble=512, time=0.5f0, dt=0.002f0,
        gamma=1.0f0, temperature=1.0f0, kappa=0.340340f0, lambda=1.0f0,
        noise_seed=seed, source_path=abspath(source), generation_seconds=seconds,
        max_abs_conservation_drift=drift)
println("L64 Model B t=0.5: $(round(seconds; digits=3)) s, max|ΔM|=$drift")
'
else
    echo "Using existing $reference"
fi

conditions=(
    "$PWD/training_data/modelB_L8_Ne51200_hmc_l1_kappa_0.340340_t0_seed2026080601.jld2"
    "$PWD/training_data/modelB_L16_Ne51200_hmc_l1_kappa_0.340340_t0_seed2026080501.jld2"
    "$PWD/training_data/modelB_L32_Ne51200_hmc_l1_kappa_0.340340_t0_seed2026080101.jld2"
    "$PWD/runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip/samples/hmc_L64_N512_seed2026080201/canonical_L64_N512_k0p340340_l1_seed2026080201.jld2"
)
lattices=(8 16 32 64)
indices=(50688:51200 50688:51200 50688:51200 0:512)

sample_model() {
    checkpoint=$1
    tag=$2
    model=$3
    seed_base=$4
    for i in 2 3; do
        lattice=${lattices[$i]}
        output="$OUT/samples/$tag/$model/L$lattice.npz"
        if [[ -e "$output" && -e "${output%.npz}.json" ]]; then
            echo "Using existing $output"
            continue
        fi
        TQDM_DISABLE=1 python -u -m solver.training.sample_modelB_cond \
            --checkpoint "$checkpoint" \
            --cond_path "${conditions[$i]}" --cond_indices "${indices[$i]}" \
            --num_steps 2000 --schedule log --batch 512 \
            --seed $((seed_base + lattice)) --output "$output"
    done
}

sample_model \
    "$PWD/runs/modelBcond_L32_k0.34034_l1.0_t0.5_ncsnpp_hmc_noskip/models/epoch=10000.ckpt" \
    t0p5 single 2026080900
sample_model \
    "$PWD/runs/modelBcond_Lmulti8-16-32_k0.34034_l1.0_t0.5_ncsnpp_hmc_noskip/models/epoch=10000.ckpt" \
    t0p5 mixed 2026080900
sample_model \
    "$PWD/runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip/models/epoch=10000.ckpt" \
    t10 single 2026081000
sample_model \
    "$PWD/runs/modelBcond_Lmulti8-16-32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip/models/epoch=10000.ckpt" \
    t10 mixed 2026081000

TQDM_DISABLE=1 PYTHONPATH=solver python - <<'PY'
from pathlib import Path
from training import sample_modelB_chain as chain

root = Path.cwd()
chain.TIMES = range(10, 201, 10)
chain.SEEDS = range(2026080801, 2026080821)
chain.BATCH = 512
chain.NUM_STEPS = 2000
chain.SCHEDULE = "log"
chain.main([
    "--checkpoint", str(root / "runs/modelBcond_Lmulti8-16-32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip/models/epoch=10000.ckpt"),
    "--input", str(root / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip/samples/hmc_L64_N512_seed2026080201/hmc_ep10000_L64_t0_N512.npz"),
    "--output_dir", str(root / "runs/multiL_ep10000_reverse_analysis/samples/t10/mixed_L64_chain_t200"),
])
PY

python -u analysis/compare_multil_reverse.py
echo "Finish: $(date '+%F %T')"
