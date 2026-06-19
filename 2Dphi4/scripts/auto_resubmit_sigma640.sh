#!/bin/bash
# Auto-resubmit wrapper. Run on the LOGIN node, inside tmux/screen:
#
#   tmux new -s resubmit_sigma640
#   bash /work/gw97/w97000/project/DM/2Dphi4/scripts/auto_resubmit_sigma640.sh
#   # detach: Ctrl-b d  ;  reattach: tmux attach -t resubmit_sigma640
#
# Each iteration uses `expect` to spawn `qsub -I` with a real pty (PBS Pro
# rejects non-TTY stdin/stdout for interactive jobs), drives the resulting
# shell to run inner_train_sigma640.sh, then exits. The outer loop relaunches
# until inner_train writes done.marker.

set -uo pipefail

WORK_DIR=/work/gw97/w97000/project/DM/2Dphi4
RUN_DIR=$WORK_DIR/runs/phi4_L128_k0.28_l0.022_ncsnpp_sigma640
DONE_MARKER=$RUN_DIR/done.marker
INNER_SCRIPT=$WORK_DIR/scripts/inner_train_sigma640.sh
LOG_DIR=$WORK_DIR/logs/auto_resubmit_sigma640

MAX_ROUNDS=${MAX_ROUNDS:-100}    # safety cap; 100 * 2h = 200h
COOLDOWN_SEC=${COOLDOWN_SEC:-15} # short pause between rounds

if ! command -v expect >/dev/null 2>&1; then
    echo "[wrapper] FATAL: 'expect' not found on this host. Install it or load a module."
    exit 127
fi

mkdir -p "$LOG_DIR"

for round in $(seq 1 "$MAX_ROUNDS"); do
    if [ -f "$DONE_MARKER" ]; then
        echo "[wrapper] Found $DONE_MARKER -- training complete, exiting."
        exit 0
    fi

    ts=$(date +%Y%m%d_%H%M%S)
    log_file="$LOG_DIR/round_${round}_${ts}.log"
    echo "===================================================================="
    echo "[wrapper] Round $round  start=$ts  log=$log_file"
    echo "===================================================================="

    # Spawn qsub -I under expect so its stdin/stdout are a real pty.
    # After PBS prints 'qsub: job <id> ready', send commands to the shell.
    # `\$result` is escaped so expect (not bash) sees $result.
    expect <<EOF 2>&1 | tee "$log_file"
log_user 1
set timeout -1
spawn qsub -I \
    -l select=1 \
    -q interact-g \
    -W group_list=gw97 \
    -l walltime=02:00:00 \
    -N dm_resubmit_sigma640

expect {
    -re "qsub: job .* ready" {}
    -re "qsub: job .* completed" { exit 0 }
    timeout { puts "[expect] timeout waiting for job ready"; exit 2 }
    eof     { puts "[expect] eof before job ready"; exit 2 }
}

# give the remote shell a moment to initialize
sleep 3

send -- "bash $INNER_SCRIPT\r"
send -- "exit\r"

expect eof
catch wait result
exit [lindex \$result 3]
EOF
    rc=${PIPESTATUS[0]}
    echo "[wrapper] Round $round expect/qsub returned rc=$rc"

    if [ -f "$DONE_MARKER" ]; then
        echo "[wrapper] done.marker appeared during round $round, exiting."
        exit 0
    fi

    echo "[wrapper] Cooling down ${COOLDOWN_SEC}s before next round..."
    sleep "$COOLDOWN_SEC"
done

echo "[wrapper] MAX_ROUNDS=$MAX_ROUNDS exhausted. Stopping."
exit 1
