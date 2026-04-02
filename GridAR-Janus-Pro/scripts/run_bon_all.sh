#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for category in color shape texture spatial non_spatial complex numeracy 3d_spatial; do
  "${SCRIPT_DIR}/run_bon_one.sh" "${category}"
done
