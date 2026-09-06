#!/bin/bash
# Create the log directory tree for every run script.
#
# SLURM does NOT create missing --output/--error directories: a job whose leaf
# log dir is absent dies immediately with a file-open error. Run this after
# adding new run scripts (or a new region/year), before sbatch:
#
#   bash scripts/maintenance/sync_log_dirs.sh
#
# Idempotent. Pass --dry-run to only list what would be created.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../lib/log_layout.sh"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

created=0
existing=0
while read -r script; do
    dir="$(log_dir_for_script "$script")" || { echo "WARN: cannot derive log dir for $script" >&2; continue; }
    if [ -d "$dir" ]; then
        existing=$((existing + 1))
    else
        created=$((created + 1))
        echo "create ${dir#"$REPO_ROOT"/}"
        [ "$DRY_RUN" -eq 1 ] || mkdir -p "$dir"
    fi
done < <(all_run_scripts)

echo "---"
echo "leaf log dirs: $existing existing, $created $([ "$DRY_RUN" -eq 1 ] && echo 'missing (dry run)' || echo created)"
