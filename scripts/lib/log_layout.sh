#!/bin/bash
# Single source of truth for the SLURM log directory layout.
#
#   log/{family}/{model}/{region}/{year}/{jobid}.out
#   log/{family}/{model}/{region}/{year}/{jobid}.err
#
# family/model come from the run-script directory (scripts/run/{family}/{model}/),
# region/year from the run-script filename (run_{slug}_{region}_{year}.sh).
# Everything that creates, rewrites or searches logs sources this file so the
# convention lives in exactly one place.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_ROOT="$REPO_ROOT/log"
RUN_ROOT="$REPO_ROOT/scripts/run"

VALID_REGIONS="sd gba gla ca"

# parse_run_script <path-to-run-script>
# On success sets LL_FAMILY / LL_MODEL / LL_REGION / LL_YEAR and returns 0.
parse_run_script() {
    local script="$1"
    local rel base stem
    # accept relative paths too
    case "$script" in /*) ;; *) script="$(cd "$(dirname "$script")" 2>/dev/null && pwd)/$(basename "$script")" ;; esac
    rel="${script#"$RUN_ROOT"/}"
    [ "$rel" = "$script" ] && return 1          # not under scripts/run/

    LL_FAMILY="${rel%%/*}"
    rel="${rel#*/}"
    LL_MODEL="${rel%%/*}"
    [ -n "$LL_FAMILY" ] && [ -n "$LL_MODEL" ] || return 1

    base="$(basename "$script" .sh)"
    stem="${base#run_}"
    LL_YEAR="${stem##*_}"
    stem="${stem%_*}"
    LL_REGION="${stem##*_}"

    [[ "$LL_YEAR" =~ ^[0-9]{4}$ ]] || return 1
    case " $VALID_REGIONS " in *" $LL_REGION "*) ;; *) return 1 ;; esac
    return 0
}

# log_dir_for_script <path-to-run-script>  ->  absolute leaf log directory
log_dir_for_script() {
    parse_run_script "$1" || return 1
    printf '%s/%s/%s/%s/%s\n' "$LOG_ROOT" "$LL_FAMILY" "$LL_MODEL" "$LL_REGION" "$LL_YEAR"
}

# all_run_scripts  ->  every run script, one per line
all_run_scripts() {
    find "$RUN_ROOT" -mindepth 3 -maxdepth 3 -name 'run_*.sh' -type f | sort
}
