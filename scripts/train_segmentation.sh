#!/bin/bash
set -e

echo "Training segmentation model..."
python -m biohub_tracking.train --model segmentation "$@"
