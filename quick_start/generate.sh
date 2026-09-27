#!/usr/bin/env bash

set -euo pipefail

#------------------------------
SRUN_ARGS="--ntasks=$SLURM_NNODES --ntasks-per-node=1"
XDG_CACHE_HOME=${SLURM_TMPDIR}/.cache
XDG_CONFIG_HOME=${SLURM_TMPDIR}/.config
TRITON_CACHE_DIR=${SLURM_TMPDIR}/.cache/triton
TMPDIR=${SLURM_TMPDIR}/.cache/tmp
SGLANG_CACHE_DIR=${SLURM_TMPDIR}/sglang_cache
SGLANG_JIT_CACHE_DIR=${SLURM_TMPDIR}/sglang_cache_jit
VLLM_CACHE_ROOT=${SLURM_TMPDIR}/vllm_cache
VLLM_CONFIG_ROOT=${SLURM_TMPDIR}/vllm_config
VLLM_ASSETS_CACHE=${SLURM_TMPDIR}/assets_cache
FLASHINFER_WORKSPACE_BASE=${SLURM_TMPDIR}/flashinfer
TORCHINDUCTOR_CACHE_DIR=${SLURM_TMPDIR}/torchindictor
SGLANG_DG_CACHE_DIR=${SLURM_TMPDIR}/sglang_dg
SGLANG_TORCH_PROFILER_DIR=${SLURM_TMPDIR}/sglang_torch_profiler

rm -rf $SLURM_TMPDIR/*

srun $SRUN_ARGS mkdir -p ${XDG_CACHE_HOME}
srun $SRUN_ARGS mkdir -p ${XDG_CONFIG_HOME}
srun $SRUN_ARGS mkdir -p ${TRITON_CACHE_DIR}
srun $SRUN_ARGS mkdir -p ${TMPDIR}
srun $SRUN_ARGS mkdir -p ${SGLANG_CACHE_DIR}
srun $SRUN_ARGS mkdir -p ${SGLANG_JIT_CACHE_DIR}
srun $SRUN_ARGS mkdir -p ${VLLM_CACHE_ROOT}
srun $SRUN_ARGS mkdir -p ${VLLM_CONFIG_ROOT}
srun $SRUN_ARGS mkdir -p ${VLLM_ASSETS_CACHE}
srun $SRUN_ARGS mkdir -p ${FLASHINFER_WORKSPACE_BASE}
srun $SRUN_ARGS mkdir -p ${TORCHINDUCTOR_CACHE_DIR}
srun $SRUN_ARGS mkdir -p ${SGLANG_DG_CACHE_DIR}
srun $SRUN_ARGS mkdir -p ${SGLANG_TORCH_PROFILER_DIR}

cd $PROJECTS_DIR/LDF-VFI
module purge all
module load StdEnv/2023 gcc/12.3 ffmpeg/7.1.1
module load cuda/12.6
source $SCRATCH/venvs/LDF-VFI/bin/activate

# ----------------------------
# Default values
# ----------------------------
num_machines=1
num_processes=1
model_path="${HF_HUB}/LDF-VFI/transformer"
vae_path="${HF_HUB}/LDF-VFI/Wan2.1_VAE_cond_v2.pth"
data_path="assets/demo.mp4"
temporal_sf=2
output_dir="${SCRATCH}/projects/LDF-VFI"
input_fps=30
sp_size=1
sampling_steps=16

# ----------------------------
# Parse user overrides (optional)
# Any of these can be passed as --key=value
# ----------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_machines=*)  num_machines="${1#*=}"  ;;
        --num_processes=*) num_processes="${1#*=}" ;;
        --model_path=*)    model_path="${1#*=}"    ;;
        --vae_path=*)      vae_path="${1#*=}"      ;;
        --data_path=*)     data_path="${1#*=}"     ;;
        --input_fps=*)   input_fps="${1#*=}"   ;;
        --temporal_sf=*)   temporal_sf="${1#*=}"   ;;
        --output_dir=*)    output_dir="${1#*=}"    ;;
        --sp_size=*)    sp_size="${1#*=}"    ;;
        --sampling_steps=*)    sampling_steps="${1#*=}"    ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
    shift
done

# ----------------------------
# Derived argument strings
# ----------------------------
distributed_args="
    --num_machines=$num_machines
    --num_processes=$num_processes
"

model_args="
    --model_path=$model_path
    --attention_type=slide_chunk_all_block_2x1x1
    --temporal_upsample=nearest
    --num_frames=60
"

vae_args="
    --vae_type=wan2_1_cond_v2
    --vae_path=$vae_path
    --vae_batch_size=16
    --tile_min_h=256
    --tile_min_w=256
    --tile_min_t=20
    --tile_stride_h=192
    --tile_stride_w=192
    --spatial_compression_ratio=8
    --temporal_compression_ratio=4
"

generate_args="
    --sampling_steps=$sampling_steps
    --t_shift=8
"

output_fps=$(awk "BEGIN {print $input_fps * $temporal_sf}")

task_args="
    --data=$data_path
    --temporal_sf=$temporal_sf
    --output_dir=$output_dir
    --fps=$output_fps
"

performance_args="
    --sp_size=$sp_size
"

mkdir -p $output_dir
echo "output_dir: $output_dir"

accelerate launch $distributed_args generate.py \
    $model_args \
    $vae_args \
    $generate_args \
    $task_args \
    $performance_args \
    2>&1 | tee -a $output_dir/log.txt

#------------------------------


# bash $PROJECTS_DIR/LDF-VFI/quick_start/generate.sh temporal_sf=2 --data_path=/scratch/rohhs/downloads/yt-dlp/death.webm --sp_size=2 --sampling_steps=16