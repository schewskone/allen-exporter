#!/usr/bin/env bash
module load apptainer

BASE_DIR="${ALLEN_BASE_DIR:?Set ALLEN_BASE_DIR to your project root}"
DATA_DIR="$BASE_DIR/data"
PACKAGE_DIR="$BASE_DIR/allen-exporter"
OVERLAY_DIR="$PACKAGE_DIR/apptainer"
OVERLAY_IMG="$OVERLAY_DIR/${SLURM_JOB_ID:-manual}.img"
INSTANCE_NAME="allen_${SLURM_JOB_ID:-manual}"

echo "Using:"
echo "Package Dir: $PACKAGE_DIR"
echo "Data Dir: $DATA_DIR"
echo "Overlay Image: $OVERLAY_IMG"

# Create overlay (if not already exists)
if [[ ! -f "$OVERLAY_IMG" ]]; then
  echo "Creating overlay image..."
  apptainer overlay create --sparse --size 1024 "$OVERLAY_IMG"
fi

# Start instance with overlay
apptainer instance start --contain \
  --overlay "$OVERLAY_IMG" \
  --bind "$PACKAGE_DIR":$HOME/package \
  --bind "$DATA_DIR":$HOME/data \
  "$PACKAGE_DIR/apptainer/allen_exporter.sif" \
  "$INSTANCE_NAME"

apptainer exec instance://$INSTANCE_NAME bash -c "pip install --no-user -e $HOME/package"
apptainer exec instance://"$INSTANCE_NAME" python $HOME/package/src/run_export.py

apptainer instance stop "$INSTANCE_NAME"

rm "$OVERLAY_IMG"
