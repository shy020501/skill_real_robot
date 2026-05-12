#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 6 ]]; then
    echo "Usage: $0 {quest|max|avg|avg_max|conv} {cuda:N|cpu} {state|force_history|state_force_history|all} [data_prefix] [state|force_history] [masked|unmasked]"
    echo "       $0 {quest|max|avg|avg_max|conv} {cuda:N|cpu} {state|force_history|state_force_history|all} [state|force_history] [masked|unmasked]"
    exit 1
fi

variant_key="$1"
device="$2"
prior_key="$3"
data_prefix="/NHNHOME/WORKSPACE/0226010443_A/seunghyo/real_robot/demos"
ft_source="state"
mask_mode="masked"
shift 3

while [[ $# -gt 0 ]]; do
    case "$1" in
        state|force_history)
            ft_source="$1"
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

case "${prior_key}" in
    state|force_history|state_force_history|all)
        ;;
    *)
        echo "Unknown prior '${prior_key}'. Expected one of: state, force_history, state_force_history, all"
        exit 1
        ;;
esac

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

case "${mask_mode}" in
    masked)
        variant_prefix="masked_"
        use_threshold_mask=true
        ;;
    unmasked)
        variant_prefix=""
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

if [[ ! -d "${autoencoder_checkpoint_dir}" ]]; then
    echo "Autoencoder checkpoint directory does not exist: ${autoencoder_checkpoint_dir}"
    echo "Run train_real_robot.sh first, or check variant/ft_source/data_prefix arguments."
    exit 1
fi

if [[ "${prior_key}" == "all" ]]; then
    prior_tasks=(state force_history state_force_history)
else
    prior_tasks=("${prior_key}")
fi

echo "[prior-only] variant=${variant}"
echo "[prior-only] mask_mode=${mask_mode}"
echo "[prior-only] checkpoint=${autoencoder_checkpoint_dir}"
echo "[prior-only] tasks=${prior_tasks[*]}"

for task_key in "${prior_tasks[@]}"; do
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
