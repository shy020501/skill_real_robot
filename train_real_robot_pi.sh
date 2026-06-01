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
    echo "Usage: $0 {quest|max|avg|avg_max|conv} {cuda:N|cpu} [data_prefix] [10hz|100hz] [masked|unmasked] [state|left_force_history|left_state_history|right_force_history|right_state_history|all]..."
    exit 1
fi

variant_key="$1"
device="$2"
data_prefix="/NHNHOME/WORKSPACE/0226010443_A/seunghyo/real_robot/demos"
ft_rate="10hz"
mask_mode="unmasked"
lowdim_modalities=()
shift 2

all_lowdim_modalities=(state left_force_history left_state_history right_force_history right_state_history)

is_lowdim_modality() {
    case "$1" in
        state|left_force_history|left_state_history|right_force_history|right_state_history|all)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

lowdim_dim() {
    case "$1" in
        state)
            echo 13
            ;;
        left_force_history|right_force_history)
            echo 60
            ;;
        left_state_history|right_state_history)
            echo 130
            ;;
        *)
            echo "Unknown lowdim modality '$1'" >&2
            exit 1
            ;;
    esac
}

join_by() {
    local IFS="$1"
    shift
    echo "$*"
}

lowdim_override() {
    local entries=()
    local modality
    for modality in "$@"; do
        entries+=("${modality}:$(lowdim_dim "${modality}")")
    done
    local joined
    joined=$(join_by "," "${entries[@]}")
    echo "+task.shape_meta.observation.lowdim={${joined}}"
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
            if is_lowdim_modality "$1"; then
                if [[ "$1" == "all" ]]; then
                    lowdim_modalities=("${all_lowdim_modalities[@]}")
                else
                    lowdim_modalities+=("$1")
                fi
            else
                data_prefix="$1"
            fi
            ;;
    esac
    shift
done

if [[ ${#lowdim_modalities[@]} -eq 0 ]]; then
    lowdim_modalities=(state)
fi

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
lowdim_name=$(join_by "_" "${lowdim_modalities[@]}")
lowdim_arg=$(lowdim_override "${lowdim_modalities[@]}")
clear_lowdim_arg="~task.shape_meta.observation.lowdim"

echo "[stage0:pi] ${variant} mask_mode=${mask_mode} pi_stats=${pi_stats_path}"
python train.py --config-name=train_autoencoder.yaml \
    "algo=${algo}" \
    "variant_name=${variant}" \
    "task=real_robot_pi" \
    "exp_name=${autoencoder_exp_name}" \
    "logging.group=autoencoder_${run_key}" \
    "${clear_lowdim_arg}" \
    "${lowdim_arg}" \
    "${common_args[@]}" \
    "${extra_args[@]}"

echo "[stage1:pi] modalities=${lowdim_modalities[*]}"
python train.py --config-name=train_prior.yaml \
    "algo=${algo}" \
    "variant_name=${variant}" \
    "task=real_robot_pi" \
    "exp_name=lowdim_${lowdim_name}_${run_key}" \
    "logging.group=${lowdim_name}_${run_key}" \
    "checkpoint_path=${autoencoder_checkpoint_dir}" \
    "${clear_lowdim_arg}" \
    "${lowdim_arg}" \
    "${common_args[@]}" \
    "${extra_args[@]}"
