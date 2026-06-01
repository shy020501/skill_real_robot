#!/usr/bin/env bash
set -euo pipefail

# Keep startup libraries from briefly fanning out across all CPU cores.
# Override these from the shell if a run needs more CPU-side throughput.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

usage() {
    echo "Usage: $0 [conv|max|avg|avg_max] {cuda:N|cpu} [data_prefix] [10hz] [masked|unmasked] [task-wise|task-agnostic]"
    echo "       $0 {cuda:N|cpu} [data_prefix] [10hz] [masked|unmasked] [task-wise|task-agnostic]"
}

is_variant() {
    case "$1" in
        conv|max|avg|avg_max)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

if [[ $# -lt 1 || $# -gt 6 ]]; then
    usage
    exit 1
fi

variant_key="conv"
if is_variant "$1"; then
    if [[ $# -lt 2 ]]; then
        usage
        exit 1
    fi
    variant_key="$1"
    device="$2"
    shift 2
else
    device="$1"
    shift 1
fi

data_prefix="/NHNHOME/WORKSPACE/0226010443_A/seunghyo/real_robot/demos"
task_agnostic_norm_stats_path="/NHNHOME/WORKSPACE/0226010443_A/seunghyo/real_robot/demos/lowdim_stats.json"
ft_label="10hz"
ft_source="state"
norm_stats_key="force"
mask_mode="unmasked"
norm_mode="task-agnostic"

while [[ $# -gt 0 ]]; do
    case "$1" in
        10hz)
            ;;
        100hz)
            echo "force reconstruction autoencoder is fixed to 10hz state force; 100hz is not supported."
            exit 1
            ;;
        masked|unmasked)
            mask_mode="$1"
            ;;
        task-wise|task_wise|taskwise)
            norm_mode="task-wise"
            ;;
        task-agnostic|task_agnostic|agnostic)
            norm_mode="task-agnostic"
            ;;
        conv|max|avg|avg_max)
            echo "Variant must be the first argument when provided. Got variant-like argument after device: $1"
            exit 1
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

case "${norm_mode}" in
    task-wise)
        norm_label=""
        norm_stats_path=""
        ;;
    task-agnostic)
        norm_label="_task_agnostic_norm"
        norm_stats_path="${task_agnostic_norm_stats_path}"
        if [[ ! -f "${norm_stats_path}" ]]; then
            echo "Task-agnostic lowdim stats file does not exist: ${norm_stats_path}"
            exit 1
        fi
        ;;
    *)
        echo "Unknown norm mode '${norm_mode}'. Expected one of: task-wise, task-agnostic"
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

algo="quest_ft_adaln"
algo_name="quest_force_recon"
variant="${variant_prefix}block_32_ds_4_force_recon_ft_${variant_key}"
run_key="force_recon_${variant_key}_${ft_label}${norm_label}"

extra_args=(
    "algo.name=${algo_name}"
    "algo.ft_downsample_mode=${variant_key}"
    "algo.dataset.ft_config.ft_source=${ft_source}"
    "algo.dataset.ft_config.use_threshold_mask=${use_threshold_mask}"
    "algo.policy.action_target_key=action_targets"
    "algo.policy.autoencoder.input_action_dim=7"
    "algo.policy.autoencoder.output_action_dim=13"
    "task.shape_meta.action_dim=13"
    "+task.dataset.action_target_mode=action_force_norm"
    "algo.dataset.ft_shift=0"
    "algo.dataset.ft_config.ema_alpha=null"
)

if [[ "${norm_mode}" == "task-agnostic" ]]; then
    extra_args+=("algo.dataset.ft_config.norm_stats_path=${norm_stats_path}")
    extra_args+=("algo.dataset.ft_config.norm_stats_key=${norm_stats_key}")
fi

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

echo "[stage0-only:force-recon] variant=${variant}"
echo "[stage0-only:force-recon] force_source=${ft_label}:${ft_source}"
echo "[stage0-only:force-recon] mask_mode=${mask_mode}"
echo "[stage0-only:force-recon] norm_mode=${norm_mode}"
echo "[stage0-only:force-recon] target=action(7)+state_force_torque(6)"
echo "[stage0-only:force-recon] checkpoint=${autoencoder_checkpoint_dir}"
python train.py --config-name=train_autoencoder.yaml \
    "algo=${algo}" \
    "variant_name=${variant}" \
    "task=real_robot_state" \
    "exp_name=${autoencoder_exp_name}" \
    "logging.group=autoencoder_${run_key}" \
    "${common_args[@]}" \
    "${extra_args[@]}"
