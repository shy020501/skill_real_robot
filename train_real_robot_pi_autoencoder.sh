#!/usr/bin/env bash
set -euo pipefail

# Keep startup libraries from briefly fanning out across all CPU cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 {quest|max|avg|avg_max|conv} {cuda:N|cpu} [data_prefix] [10hz|100hz] [masked|unmasked]"
    exit 1
fi

variant_key="$1"
device="$2"
data_prefix="/NHNHOME/WORKSPACE/0226010443_A/seunghyo/real_robot/demos"
ft_rate="10hz"
mask_mode="unmasked"
shift 2

wait_for_gpu_if_needed() {
    local gpu_index
    local used_mb
    local pids
    local poll_seconds="${GPU_WAIT_POLL_SECONDS:-60}"
    local free_memory_threshold_mb="${GPU_FREE_MEMORY_THRESHOLD_MB:-100}"

    if [[ "${device}" == "cpu" ]]; then
        return 0
    fi

    if [[ ! "${device}" =~ ^cuda:([0-9]+)$ ]]; then
        echo "[gpu-wait] Skip GPU availability check for device=${device}"
        return 0
    fi

    gpu_index="${BASH_REMATCH[1]}"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[gpu-wait] nvidia-smi not found; cannot check ${device}"
        return 0
    fi

    echo "[gpu-wait] Waiting for ${device} to be free..."
    while true; do
        used_mb="$(nvidia-smi -i "${gpu_index}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")"
        pids="$(nvidia-smi -i "${gpu_index}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed "/^[[:space:]]*$/d" || true)"

        if [[ -z "${pids}" && "${used_mb}" -le "${free_memory_threshold_mb}" ]]; then
            echo "[gpu-wait] ${device} is free: memory_used=${used_mb}MiB"
            return 0
        fi

        echo "[gpu-wait] ${device} busy: memory_used=${used_mb}MiB pids=${pids:-none}; retry in ${poll_seconds}s"
        sleep "${poll_seconds}"
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        10hz|100hz)
            ft_rate="$1"
            ;;
        masked|unmasked)
            mask_mode="$1"
            ;;
        *)
            data_prefix="$1"
            ;;
    esac
    shift
done

if [[ ! -d "${data_prefix}" ]]; then
    echo "Data prefix does not exist: ${data_prefix}"
    exit 1
fi

pi_stats_path="${data_prefix}/pi_stats.json"
if [[ ! -f "${pi_stats_path}" ]]; then
    echo "[stats] ${pi_stats_path} does not exist; computing it now."
    python scripts/compute_stats_pi.py --data_prefix "${data_prefix}" --output "${pi_stats_path}"
fi

case "${ft_rate}" in
    10hz)
        ft_source="state"
        ft_label="${ft_rate}"
        ;;
    100hz)
        ft_source="right_force_history"
        ft_label="${ft_rate}"
        ;;
    *)
        echo "Unknown FT rate '${ft_rate}'. Expected one of: 10hz, 100hz"
        exit 1
        ;;
esac

case "${mask_mode}" in
    masked)
        variant_prefix="masked_"
        use_threshold_mask=true
        ;;
    unmasked)
        variant_prefix="unmasked_"
        use_threshold_mask=false
        ;;
    *)
        echo "Unknown mask mode '${mask_mode}'. Expected one of: masked, unmasked"
        exit 1
        ;;
esac

extra_args=()
run_key="${variant_key}_pi"
case "${variant_key}" in
    quest)
        algo="quest"
        algo_name="quest"
        variant="${variant_prefix}block_48_ds_4_quest_pi"
        ;;
    max|avg|avg_max|conv)
        algo="quest_ft_adaln"
        algo_name="quest_ft_adaln"
        variant="${variant_prefix}block_48_ds_4_ft_${variant_key}_pi"
        extra_args+=("algo.ft_downsample_mode=${variant_key}")
        extra_args+=("algo.dataset.ft_config.ft_source=${ft_source}")
        extra_args+=("algo.dataset.ft_config.use_threshold_mask=${use_threshold_mask}")
        if [[ "${variant_key}" == "conv" && "${ft_rate}" == "100hz" ]]; then
            extra_args+=("algo.ft_conv_strides=[5,4,2]")
            extra_args+=("algo.ft_conv_kernel_sizes=[9,7,5]")
        fi
        run_key="${variant_key}_${ft_label}_pi"
        ;;
    *)
        echo "Unknown variant '${variant_key}'. Expected one of: quest, max, avg, avg_max, conv"
        exit 1
        ;;
esac

common_args=(
    "training.use_tqdm=false"
    "training.save_all_checkpoints=true"
    "training.use_amp=false"
    "train_dataloader.persistent_workers=true"
    "train_dataloader.num_workers=4"
    "train_dataloader.multiprocessing_context=fork"
    "make_unique_experiment_dir=false"
    "algo.skill_block_size=48"
    "algo.downsample_factor=4"
    "algo.encoder.lowdim.encoder_type_by_modality.left_force_history=mlp"
    "algo.encoder.lowdim.encoder_type_by_modality.right_force_history=mlp"
    "seed=0"
    "data_prefix=${data_prefix}"
    "task.instruction_path=${data_prefix}/instructions.json"
    "+algo.dataset.pi_stats_path=${pi_stats_path}"
    "device=${device}"
)

autoencoder_exp_name="lowdim_autoencoder_${run_key}"
autoencoder_checkpoint_dir="./experiments/real_robot/REAL_ROBOT_MULTI/${algo_name}/${autoencoder_exp_name}/${variant}/0/stage_0"

echo "[stage0-only:pi] variant=${variant}"
echo "[stage0-only:pi] mask_mode=${mask_mode}"
echo "[stage0-only:pi] pi_stats=${pi_stats_path}"
echo "[stage0-only:pi] checkpoint=${autoencoder_checkpoint_dir}"
wait_for_gpu_if_needed
python train.py --config-name=train_autoencoder.yaml \
    "algo=${algo}" \
    "variant_name=${variant}" \
    "task=real_robot_state_pi" \
    "exp_name=${autoencoder_exp_name}" \
    "logging.group=autoencoder_${run_key}" \
    "${common_args[@]}" \
    "${extra_args[@]}"
