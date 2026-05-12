#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
    echo "Usage: $0 {quest|max|avg|avg_max|conv} {cuda:N|cpu} [data_prefix] [10hz|100hz] [masked|unmasked]"
    echo "       $0 {quest|max|avg|avg_max|conv} {cuda:N|cpu} [10hz|100hz] [masked|unmasked]"
    exit 1
fi

variant_key="$1"
device="$2"
data_prefix="/NHNHOME/WORKSPACE/0226010443_A/seunghyo/real_robot/demos"
ft_rate="10hz"
mask_mode="unmasked"
shift 2

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

case "${ft_rate}" in
    10hz)
        ft_source="state"
        ft_label="${ft_rate}"
        ;;
    100hz)
        ft_source="force_history"
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
run_key="${variant_key}"
case "${variant_key}" in
    quest)
        algo="quest"
        algo_name="quest"
        variant="${variant_prefix}block_32_ds_4_quest"
        ;;
    max|avg|avg_max|conv)
        algo="quest_ft_adaln"
        algo_name="quest_ft_adaln"
        variant="${variant_prefix}block_32_ds_4_ft_${variant_key}"
        extra_args+=("algo.ft_downsample_mode=${variant_key}")
        extra_args+=("algo.dataset.ft_config.ft_source=${ft_source}")
        extra_args+=("algo.dataset.ft_config.use_threshold_mask=${use_threshold_mask}")
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

echo "[stage0] ${variant} mask_mode=${mask_mode}"
python train.py --config-name=train_autoencoder.yaml \
    "algo=${algo}" \
    "variant_name=${variant}" \
    "task=real_robot_state" \
    "exp_name=${autoencoder_exp_name}" \
    "logging.group=autoencoder_${run_key}" \
    "${common_args[@]}" \
    "${extra_args[@]}"

for task_key in state force_history state_force_history; do
    case "${task_key}" in
        state)
            task_config="real_robot_state"
            ;;
        force_history)
            task_config="real_robot_force_history"
            ;;
        state_force_history)
            task_config="real_robot_state_force_history"
            ;;
    esac

    echo "[stage1] ${variant} ${task_key}"
    python train.py --config-name=train_prior.yaml \
        "algo=${algo}" \
        "variant_name=${variant}" \
        "task=${task_config}" \
        "exp_name=lowdim_${task_key}_${run_key}" \
        "logging.group=${task_key}_${run_key}" \
        "checkpoint_path=${autoencoder_checkpoint_dir}" \
        "${common_args[@]}" \
        "${extra_args[@]}"
done
