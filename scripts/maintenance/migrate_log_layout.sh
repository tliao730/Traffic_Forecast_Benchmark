#!/bin/bash
# One-shot migration to the nested log layout:
#
#   old:  log/{family}/{jobid}_run_{model}_{region}_{year}.out
#   new:  log/{family}/{model}/{region}/{year}/{jobid}.out
#
# Does two things:
#   1. rewrites every run script's #SBATCH --output/--error to the new path
#   2. moves existing flat log files into the new tree
#
# Safe to re-run: step 1 is idempotent, step 2 skips files already nested and
# never overwrites an existing destination. Unrecognised filenames go to
# log/{family}/_misc/ instead of being left behind.
#
#   bash scripts/maintenance/migrate_log_layout.sh --dry-run
#   bash scripts/maintenance/migrate_log_layout.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../lib/log_layout.sh"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
run() { if [ "$DRY_RUN" -eq 1 ]; then echo "  would: $*"; else "$@"; fi; }

# --- 1. rewrite SBATCH log paths -------------------------------------------
echo "== rewriting #SBATCH --output/--error in run scripts =="
rewritten=0
unchanged=0
while read -r script; do
    dir="$(log_dir_for_script "$script")" || { echo "WARN: skipping $script (cannot derive log dir)" >&2; continue; }
    out="$dir/%j.out"
    err="$dir/%j.err"
    if grep -qxF "#SBATCH --output=$out" "$script" && grep -qxF "#SBATCH --error=$err" "$script"; then
        unchanged=$((unchanged + 1))
        continue
    fi
    rewritten=$((rewritten + 1))
    echo "  ${script#"$REPO_ROOT"/}"
    if [ "$DRY_RUN" -eq 0 ]; then
        python3 - "$script" "$out" "$err" <<'PY'
import re, sys
path, out, err = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f:
    text = f.read()
text, n_out = re.subn(r'(?m)^#SBATCH --output=.*$', '#SBATCH --output=' + out, text)
text, n_err = re.subn(r'(?m)^#SBATCH --error=.*$',  '#SBATCH --error='  + err, text)
if n_out != 1 or n_err != 1:
    sys.exit(f'{path}: expected 1 --output and 1 --error line, found {n_out}/{n_err}')
with open(path, 'w') as f:
    f.write(text)
PY
    fi
done < <(all_run_scripts)
echo "  $rewritten rewritten, $unchanged already correct"

# --- 2. move existing flat logs into the tree ------------------------------
echo "== moving existing log files =="
declare -A DEST
while read -r script; do
    dir="$(log_dir_for_script "$script")" || continue
    DEST["$(basename "$script" .sh)"]="$dir"
done < <(all_run_scripts)

# historical name changes: old log stem prefix -> current run-script prefix
rename_stem() {
    local stem="$1"
    stem="${stem/#run_chronos_/run_chronos_bolt_}"   # chronos -> chronos_bolt
    printf '%s\n' "$stem"
}

# Logs of jobs still in the queue are left where they are: SLURM already has
# them open at the old flat path, and their queued follow-ups were submitted
# with the old script copy.
ACTIVE_JOBS=" $(squeue -u "$USER" -h -o %i 2>/dev/null | tr '\n' ' ')"

moved=0
misc=0
skipped=0
active=0
while read -r file; do
    rel="${file#"$LOG_ROOT"/}"
    family="${rel%%/*}"
    name="$(basename "$file")"
    # already nested (family/model/region/year/jobid.ext) -> leave alone
    depth=$(awk -F/ '{print NF}' <<<"$rel")
    if [ "$depth" -gt 2 ]; then
        skipped=$((skipped + 1))
        continue
    fi
    jobid="${name%%_*}"
    if [[ "$jobid" =~ ^[0-9]+$ ]] && [[ "$ACTIVE_JOBS" == *" $jobid "* ]]; then
        echo "  KEEP (job $jobid still queued/running): $rel"
        active=$((active + 1))
        continue
    fi
    stem="${name#*_}"
    ext="${stem##*.}"
    stem="${stem%.*}"
    stem="$(rename_stem "$stem")"
    dest="${DEST[$stem]:-}"
    if [ -n "$dest" ] && [[ "$jobid" =~ ^[0-9]+$ ]]; then
        target="$dest/$jobid.$ext"
        moved=$((moved + 1))
    else
        target="$LOG_ROOT/$family/_misc/$name"
        misc=$((misc + 1))
    fi
    if [ -e "$target" ]; then
        echo "  SKIP (exists): ${target#"$LOG_ROOT"/}" >&2
        continue
    fi
    run mkdir -p "$(dirname "$target")"
    run mv "$file" "$target"
done < <(find "$LOG_ROOT" -type f ! -name '.gitkeep' | sort)
echo "  $moved matched, $misc -> _misc, $skipped already nested, $active left in place (active jobs)"

# --- 3. make sure every leaf dir exists ------------------------------------
echo "== creating any missing leaf dirs =="
if [ "$DRY_RUN" -eq 1 ]; then
    bash "$REPO_ROOT/scripts/maintenance/sync_log_dirs.sh" --dry-run | tail -3
else
    bash "$REPO_ROOT/scripts/maintenance/sync_log_dirs.sh" | tail -3
fi
