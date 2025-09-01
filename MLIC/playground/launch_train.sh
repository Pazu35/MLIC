#!/bin/bash

# MLIC++ Training Launch Script
# This script allows you to easily configure and launch training with different parameters

# Default values
CFG_N=192
CFG_M=320
LAMBDA=0.0018
LOAD_DM=false
DM_PATH="/Odyssey/private/o23gauvr/code/FASCINATION/pickle/dm_enatl_mean_std_along_depth_4_157_196_256.pkl"
EPOCHS=34000
BATCH_SIZE=32
LR=1e-4
GPU_ID="0"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n, --cfg_N N          Model config N parameter (default: $CFG_N)"
    echo "  -m, --cfg_M M          Model config M parameter (default: $CFG_M)"
    echo "  -l, --lambda LAMBDA    Lambda value for rate-distortion loss (default: $LAMBDA)"
    echo "  --load_dm              Load datamodule from pickle file"
    echo "  --dm_path PATH         Path to datamodule pickle file (default: $DM_PATH)"
    echo "  -e, --epochs EPOCHS    Number of epochs (default: $EPOCHS)"
    echo "  -b, --batch_size SIZE  Batch size (default: $BATCH_SIZE)"
    echo "  --lr RATE              Learning rate (default: $LR)"
    echo "  -g, --gpu_id ID        GPU ID (default: $GPU_ID)"
    echo "  -h, --help             Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -n 256 -m 384 -l 0.0035                    # Basic parameter change"
    echo "  $0 --load_dm --dm_path /path/to/datamodule.pkl # Load existing datamodule"
    echo "  $0 -n 192 -m 320 -e 1000 -b 16               # Quick test run"
    echo ""
    echo "Common lambda values: 0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--cfg_N)
            CFG_N="$2"
            shift 2
            ;;
        -m|--cfg_M)
            CFG_M="$2"
            shift 2
            ;;
        -l|--lambda)
            LAMBDA="$2"
            shift 2
            ;;
        --load_dm)
            LOAD_DM=true
            shift
            ;;
        --dm_path)
            DM_PATH="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        -g|--gpu_id)
            GPU_ID="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            usage
            exit 1
            ;;
    esac
done

# Display configuration
echo "=================================================="
echo "MLIC++ Training Configuration"
echo "=================================================="
echo "Model Config N:     $CFG_N"
echo "Model Config M:     $CFG_M"
echo "Lambda:             $LAMBDA"
echo "Load Datamodule:    $LOAD_DM"
echo "Datamodule Path:    $DM_PATH"
echo "Epochs:             $EPOCHS"
echo "Batch Size:         $BATCH_SIZE"
echo "Learning Rate:      $LR"
echo "GPU ID:             $GPU_ID"
echo "=================================================="
echo ""

# Check if datamodule file exists when load_dm is true
if [ "$LOAD_DM" = true ] && [ ! -f "$DM_PATH" ]; then
    echo "Error: Datamodule file not found at $DM_PATH"
    echo "Please check the path or create the datamodule first."
    exit 1
fi

# Confirm before starting
read -p "Do you want to start training with these settings? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Change to the correct directory
cd /Odyssey/private/o23gauvr/code/MLIC/MLIC/playground

# Build the command
CMD="python train.py"
CMD="$CMD --cfg_N $CFG_N"
CMD="$CMD --cfg_M $CFG_M"
CMD="$CMD --lambda_val $LAMBDA"
CMD="$CMD --dm_path \"$DM_PATH\""
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --lr $LR"
CMD="$CMD --gpu_id $GPU_ID"

if [ "$LOAD_DM" = true ]; then
    CMD="$CMD --load_dm"
fi

# Log the command being executed
echo "Executing command:"
echo "$CMD"
echo ""

# Execute the training
eval $CMD
