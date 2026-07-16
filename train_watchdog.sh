#!/bin/bash
# Auto-restart watchdog for the me-full training run. GPU wedges under co-tenancy
# stall the process silently (see TRAINING_DIARY.md 2026-07-08 19:xx); this loop
# kills and relaunches on death OR on a stalled log, and training resumes from the
# rolling step checkpoint (max ~1000 steps lost).
cd "$(dirname "$0")" || exit 1
LOG=${TRAIN_LOG:-train_me-full.log}
WLOG=watchdog.log
PIDFILE=.watchdog.pid
STALL_SECS=1200    # 20 min without a log write = wedged (steps print every ~20-30 s)
MIN_AGE=1500       # never judge a process younger than 25 min (compile/load phase)
MAX_RESTARTS_6H=6  # give up if wedging faster than this; humans should intervene

if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
  echo "watchdog already running (pid $(cat $PIDFILE))"; exit 1
fi
echo $$ > "$PIDFILE"
ts() { date "+%F %T"; }
echo "$(ts) watchdog started (pid $$)" >> $WLOG

RESTART_TIMES=()
relaunch() {
  NOW=$(date +%s)
  RESTART_TIMES+=("$NOW")
  RECENT=0
  for t in "${RESTART_TIMES[@]}"; do [ $((NOW - t)) -lt 21600 ] && RECENT=$((RECENT+1)); done
  if [ $RECENT -gt $MAX_RESTARTS_6H ]; then
    echo "$(ts) GIVING UP: $RECENT restarts in 6 h. Not relaunching." >> $WLOG
    rm -f "$PIDFILE"; exit 1
  fi
  echo "$(ts) relaunching (restart #${#RESTART_TIMES[@]}, $RECENT in last 6 h)" >> $WLOG
  HIP_VISIBLE_DEVICES=${TRAIN_GPU:-3} nohup setsid ./launch_train.sh > $LOG 2>&1 &
  sleep 60
}

while true; do
  sleep 300
  PID=$(pgrep -f "kin_flow.cli.train" | while read p; do
          [ "$(ps -o comm= -p "$p" 2>/dev/null)" = "python" ] && echo "$p"; done | head -1)
  if [ -z "$PID" ]; then
    echo "$(ts) training process dead; last log lines:" >> $WLOG
    tail -3 $LOG >> $WLOG
    relaunch
    continue
  fi
  AGE=$(ps -o etimes= -p "$PID" | tr -d ' ')
  IDLE=$(( $(date +%s) - $(stat -c %Y $LOG) ))
  if [ "$AGE" -gt $MIN_AGE ] && [ "$IDLE" -gt $STALL_SECS ]; then
    echo "$(ts) STALL: pid $PID up ${AGE}s, log idle ${IDLE}s; killing" >> $WLOG
    kill -9 "$PID"; sleep 20
    relaunch
  fi
done
