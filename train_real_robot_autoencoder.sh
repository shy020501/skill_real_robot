#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: $0 {quest|max|avg|avg_max|conv} {cuda:N|cpu} [data_prefix] [state|force_history]"
    echo "       $0 {quest|max|avg|avg_max|conv} {cuda:N|cpu} [state|force_history]"
    exit 1
fi

variant_key="$1"
device="$2"
data_prefix="/NHNHOME/WORKSPACE/0226010443_A/seunghyo/real_robot/demos"
ft_source="state"

if [[ $# -ge 3 ]]; then
    case "$3" in
        state|force_history)
            ft_source="$3"
            ;;
        *)
            data_prefix="$3"
            ;;
    esac
fi

if [[ $# -eq 4 ]]; then
    ft_source="$4"
fi

if [[ ! -d "${data_prefix}" ]]; then
    echo "Data prefix does not exist: ${data_prefix}"
    exit 1
fi

case "${ft_source}" in
    state)
        ft_label="10hz"
        ;;
    force_history)
        ft_label="100hz"
        ;;
    *)
        echo "Unknown ft_source '${ft_source}'. Expected one of: state, force_history"
        exit 1
        ;;
esac

extra_args=()
run_key="${variant_key}"
case "${variant_key}" in
    quest)
        algo="quest"
        algo_name="quest"
        variant="masked_block_32_ds_4_quest"
        ;;
    max|avg|avg_max|conv)
        algo="quest_ft_adaln"
        algo_name="quest_ft_adaln"
        variant="masked_block_32_ds_4_ft_${variant_key}"
        extra_args+=("algo.ft_downsample_mode=${variant_key}")
        extra_args+=("algo.dataset.ft_config.ft_source=${ft_source}")
        run_key="${variant_key}_${ft_label}"
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
    "train_dataloader.num_workers=6"
    "train_dataloader.multiprocessing_context=fork"
    "make_unique_experiment_dir=false"
    "algo.skill_block_size=32"
    "algo.downsample_factor=4"
    "seed=0"
    "data_prefix=${data_prefix}"
    "device=${device}"
)

autoencoder_exp_name="lowdim_autoencoder_${run_key}"
autoencoder_checkpoint_dir="./experiments/real_robot/REAL_ROBOT_MULTI/${algo_name}/${autoencoder_exp_name}/${variant}/0/stage_0"

echo "[stage0-only] variant=${variant}"
echo "[stage0-only] checkpoint=${autoencoder_checkpoint_dir}"
python train.py --config-name=train_autoencoder.yaml \
    "algo=${algo}" \
    "variant_name=${variant}" \
    "task=real_robot_state" \
    "exp_name=${autoencoder_exp_name}" \
    "logging.group=autoencoder_${run_key}" \
    "${common_args[@]}" \
    "${extra_args[@]}"
