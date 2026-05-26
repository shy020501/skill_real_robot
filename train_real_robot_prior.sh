#!/usr/bin/env bash
set -euo pipefail

# Keep startup libraries from briefly fanning out across all CPU cores.
# Override these from the shell if a run needs more CPU-side throughput.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 {quest|max|avg|avg_max|conv} {cuda:N|cpu} {state|left_force_history|left_state_history|right_force_history|right_state_history|all}... [data_prefix] [10hz|100hz] [masked|unmasked] [task-wise|task-agnostic]"
    exit 1
fi

variant_key="$1"
device="$2"
data_prefix="/NHNHOME/WORKSPACE/0226010443_A/seunghyo/real_robot/demos"
task_agnostic_norm_stats_path="/NHNHOME/WORKSPACE/0226010443_A/seunghyo/real_robot/demos/lowdim_stats.json"
ft_rate="10hz"
mask_mode="unmasked"
norm_mode="task-wise"
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
        task-wise|task_wise|taskwise)
            norm_mode="task-wise"
            ;;
        task-agnostic|task_agnostic|agnostic)
            norm_mode="task-agnostic"
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

case "${ft_rate}" in
    10hz)
        ft_source="state"
        ft_label="${ft_rate}"
        norm_stats_key="force"
        ;;
    100hz)
        ft_source="right_force_history"
        ft_label="${ft_rate}"
        norm_stats_key="right_force_history"
        ;;
    *)
        echo "Unknown FT rate '${ft_rate}'. Expected one of: 10hz, 100hz"
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
        if [[ "${norm_mode}" == "task-agnostic" ]]; then
            extra_args+=("algo.dataset.ft_config.norm_stats_path=${norm_stats_path}")
            extra_args+=("algo.dataset.ft_config.norm_stats_key=${norm_stats_key}")
        fi
        if [[ "${variant_key}" == "conv" && "${ft_rate}" == "100hz" ]]; then
            extra_args+=("algo.ft_conv_strides=[5,4,2]")
            extra_args+=("algo.ft_conv_kernel_sizes=[9,7,5]")
        fi
        run_key="${variant_key}_${ft_label}${norm_label}"
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
    "algo.skill_block_size=32"
    "algo.downsample_factor=4"
    "seed=0"
    "data_prefix=${data_prefix}"
    "task.instruction_path=${data_prefix}/instructions.json"
    "device=${device}"
)

autoencoder_exp_name="lowdim_autoencoder_${run_key}"
autoencoder_checkpoint_dir="./experiments/real_robot/REAL_ROBOT_MULTI/${algo_name}/${autoencoder_exp_name}/${variant}/0/stage_0"
lowdim_name=$(join_by "_" "${lowdim_modalities[@]}")
lowdim_arg=$(lowdim_override "${lowdim_modalities[@]}")
clear_lowdim_arg="~task.shape_meta.observation.lowdim"

if [[ ! -d "${autoencoder_checkpoint_dir}" ]]; then
    echo "Autoencoder checkpoint directory does not exist: ${autoencoder_checkpoint_dir}"
    echo "Run train_real_robot.sh first, or check variant/ft_source/data_prefix arguments."
    exit 1
fi

echo "[prior-only] variant=${variant}"
echo "[prior-only] mask_mode=${mask_mode}"
echo "[prior-only] norm_mode=${norm_mode}"
echo "[prior-only] checkpoint=${autoencoder_checkpoint_dir}"
echo "[prior-only] modalities=${lowdim_modalities[*]}"

echo "[stage1] ${variant} ${lowdim_name}"
python train.py --config-name=train_prior.yaml \
    "algo=${algo}" \
    "variant_name=${variant}" \
    "task=real_robot" \
    "exp_name=lowdim_${lowdim_name}_${run_key}_pretrained" \
    "logging.group=${lowdim_name}_${run_key}" \
    "checkpoint_path=${autoencoder_checkpoint_dir}" \
    "${clear_lowdim_arg}" \
    "${lowdim_arg}" \
    "${common_args[@]}" \
    "${extra_args[@]}"
