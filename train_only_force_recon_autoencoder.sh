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
    echo "Usage: $0 [quest] {cuda:N|cpu} [data_prefix] [10hz] [task-wise|task-agnostic]"
}

if [[ $# -lt 1 || $# -gt 5 ]]; then
    usage
    exit 1
fi

if [[ "$1" == "quest" ]]; then
    if [[ $# -lt 2 ]]; then
        usage
        exit 1
    fi
    shift
fi

device="$1"
shift

data_prefix="/NHNHOME/WORKSPACE/0226010443_A/seunghyo/real_robot/demos"
task_agnostic_norm_stats_path="/NHNHOME/WORKSPACE/0226010443_A/seunghyo/real_robot/demos/lowdim_stats.json"
norm_mode="task-agnostic"
downsample_factor="${DOWNSAMPLE_FACTOR:-4}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        10hz)
            ;;
        100hz)
            echo "only-force reconstruction autoencoder is fixed to 10hz state force/torque; 100hz is not supported."
            exit 1
            ;;
        task-wise|task_wise|taskwise)
            norm_mode="task-wise"
            ;;
        task-agnostic|task_agnostic|agnostic)
            norm_mode="task-agnostic"
            ;;
        quest)
            echo "The optional quest argument must be first."
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

case "${downsample_factor}" in
    1|2|4|8|16|32)
        ;;
    *)
        echo "Unsupported DOWNSAMPLE_FACTOR='${downsample_factor}'. Expected a power of two in: 1 2 4 8 16 32"
        exit 1
        ;;
esac

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

algo="quest"
algo_name="quest_only_force_recon"
variant="block_32_ds_${downsample_factor}_only_force_recon"
run_key="only_force_recon_ds_${downsample_factor}${norm_label}"

extra_args=(
    "algo.name=${algo_name}"
    "task.shape_meta.action_dim=6"
    "algo.policy.autoencoder.input_action_dim=6"
    "algo.policy.autoencoder.output_action_dim=6"
    "+task.dataset.action_input_mode=state_force_norm"
    "+task.dataset.leading_keep=32"
)

if [[ "${norm_mode}" == "task-agnostic" ]]; then
    extra_args+=("task.dataset.lowdim_stats_path=${norm_stats_path}")
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
    "algo.downsample_factor=${downsample_factor}"
    "seed=0"
    "data_prefix=${data_prefix}"
    "device=${device}"
)

autoencoder_exp_name="lowdim_autoencoder_${run_key}"
autoencoder_checkpoint_dir="./experiments/real_robot/REAL_ROBOT_MULTI/${algo_name}/${autoencoder_exp_name}/${variant}/0/stage_0"

echo "[stage0-only:only-force-recon] variant=${variant}"
echo "[stage0-only:only-force-recon] force_source=10hz:state"
echo "[stage0-only:only-force-recon] norm_mode=${norm_mode}"
echo "[stage0-only:only-force-recon] downsample_factor=${downsample_factor}"
echo "[stage0-only:only-force-recon] target=state_force_torque(6)"
echo "[stage0-only:only-force-recon] checkpoint=${autoencoder_checkpoint_dir}"
python train.py --config-name=train_autoencoder.yaml \
    "algo=${algo}" \
    "variant_name=${variant}" \
    "task=real_robot_state" \
    "exp_name=${autoencoder_exp_name}" \
    "logging.group=autoencoder_${run_key}" \
    "${common_args[@]}" \
    "${extra_args[@]}"
