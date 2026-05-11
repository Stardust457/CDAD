#!/bin/bash

CLASSES=(
    "bracket_black"
    "bracket_brown"
    "bracket_white"
    "connector"
    "metal_plate"
    "tubes"
)

# evaluate image_auroc
# VAL_MODE="image_auroc"

# evaluate pixel_auroc
VAL_MODE="pixel_auroc"

if [ "$VAL_MODE" = "pixel_auroc" ]; then
    VAL_ARGS=(--val_monitor "pixel_auroc" --log_pixel_metrics 1)
else
    VAL_ARGS=(--val_monitor "image_auroc" --log_pixel_metrics 0)
fi

COMMON_ARGS=(
    --epochs 160
    --batch_size 16
    --test_batch_size 16
    --lr 0.0005
    --lr_decay_factor 0.2
    --seed 0
    --hf_path 'vit_large_patch14_dinov2.lvd142m'
    --image_size 518
    --layers_to_extract_from '24'
    --hidden_dim 2048
    --noise_std 0.25
    --log_every_n_steps 4
    --run_type "cdad"
    --dataset_name 'mpdd'
    --wandb_entity ""
    --wandb_api_key ""
    --wandb_name "CDAD"
    --data_dir './autodl-tmp'
    --num_fake_patches -1
    --dsc_layers 1
    --dsc_heads 4
    --dsc_dropout 0.1
    --fake_feature_type 'random'
    --top_k 10
    --smoothing_sigma 16
    --smoothing_radius 18
    --shots -1
    --lr_adaptor=0.0001
)

TOTAL=${#CLASSES[@]}

echo "========================================"
echo "  MPDD "
echo "  Total number of classes: ${TOTAL} "
echo "========================================"

for i in "${!CLASSES[@]}"; do
    CLASS="${CLASSES[$i]}"
    IDX=$((i + 1))

    echo ""
    echo "----------------------------------------"
    echo "  [${IDX}/${TOTAL}] Running class: ${CLASS}"
    echo "----------------------------------------"

    python main.py \
        --normal_class "${CLASS}" \
        "${COMMON_ARGS[@]}" \
        "${VAL_ARGS[@]}"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "Class '${CLASS}' training failed (exitcode: ${EXIT_CODE}),skipping to next one."
    else
        echo ""
        echo "Class '${CLASS}' training completed."
    fi
done

echo ""
echo "========================================"
echo "  All classes have been trained. "
echo "========================================"