#!/bin/bash
# Locate SLURM logs in the nested layout log/{family}/{model}/{region}/{year}/{jobid}.{out,err}
#
#   find_log.sh                     newest 20 logs across every model
#   find_log.sh lru                 newest logs for model lru (any region/year)
#   find_log.sh lru sd 2018         narrow to region + year
#   find_log.sh -t lru sd 2018      tail -f the newest .out for that cell
#   find_log.sh -e lru sd 2018      tail -f the newest .err instead
#   find_log.sh -n 50 gnn           show 50 newest under family gnn
#
# Filters are plain substrings matched against the path, so any of
# family / model / region / year (in any order) works.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../lib/log_layout.sh"

TAIL=0
EXT="out"
LIMIT=20
while [ $# -gt 0 ]; do
    case "$1" in
        -t|--tail) TAIL=1; shift ;;
        -e|--err)  TAIL=1; EXT="err"; shift ;;
        -n)        LIMIT="$2"; shift 2 ;;
        -h|--help) sed -n '2,16p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *)         break ;;
    esac
done

mapfile -t files < <(
    find "$LOG_ROOT" -type f \( -name '*.out' -o -name '*.err' \) -printf '%T@ %p\n' \
    | sort -rn | cut -d' ' -f2-
)

for filter in "$@"; do
    keep=()
    for f in "${files[@]}"; do
        [[ "${f#"$LOG_ROOT"/}" == *"$filter"* ]] && keep+=("$f")
    done
    files=("${keep[@]+"${keep[@]}"}")
done

if [ "${#files[@]}" -eq 0 ]; then
    echo "no logs match: $*" >&2
    exit 1
fi

if [ "$TAIL" -eq 1 ]; then
    for f in "${files[@]}"; do
        if [ "${f##*.}" = "$EXT" ]; then
            echo "== ${f#"$LOG_ROOT"/} ==" >&2
            exec tail -f "$f"
        fi
    done
    echo "no .$EXT log matches: $*" >&2
    exit 1
fi

printf '%s\n' "${files[@]}" | head -n "$LIMIT" | while read -r f; do
    printf '%s  %8s  %s\n' "$(date -r "$f" '+%Y-%m-%d %H:%M')" "$(du -h "$f" | cut -f1)" "${f#"$LOG_ROOT"/}"
done
