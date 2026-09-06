#!/bin/bash
set -e

# Setup environment or run pre-flight checks here
echo "Starting BioHub Cell Tracking container..."

# Execute the main command
exec "$@"
