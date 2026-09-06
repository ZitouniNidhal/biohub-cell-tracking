#!/bin/bash
set -e

echo "Running inference..."
python -m biohub_tracking.inference "$@"
