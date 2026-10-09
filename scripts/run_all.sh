#!/usr/bin/env bash
# Everything, in dependency order:  tests -> demos -> synthetic analysis -> DICES.
#   scripts/run_all.sh          the quick path: reuses the saved benchmark data and search results
#   scripts/run_all.sh --full   also regenerates the synthetic data and reruns the search (hours)
here="$(dirname "$0")"
source "$here/_common.sh"

FULL=0
[ "${1:-}" = "--full" ] && FULL=1

"$here/tests.sh"
if [ "$FULL" = 1 ]; then
  "$here/synthetic_data.sh"
  "$here/synthetic_benchmark.sh"
fi
"$here/demos.sh"
"$here/synthetic_analysis.sh"
"$here/dices.sh"
step "all done"
