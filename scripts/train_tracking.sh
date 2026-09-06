#!/bin/bash
set -e

echo "Training tracking model..."
python -m biohub_tracking.train --model tracking "$@"
