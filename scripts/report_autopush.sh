#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/report_autopush.sh"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUNTIME_DIR="${REPORT_AUTOPUSH_DIR:-$REPO_ROOT/.git-report-autopush}"
PID_FILE="$RUNTIME_DIR/pid"
LOG_FILE="$RUNTIME_DIR/report-autopush.log"
REPORTS_PATH="${REPORT_AUTOPUSH_PATH:-outputs}"
INTERVAL_SECONDS="${REPORT_AUTOPUSH_INTERVAL_SECONDS:-1800}"
REMOTE_NAME="${REPORT_AUTOPUSH_REMOTE:-origin}"
BRANCH_NAME="${REPORT_AUTOPUSH_BRANCH:-$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)}"

usage() {
  cat <<'USAGE'
Usage: scripts/report_autopush.sh <command>

Commands:
  run-once   Stage, commit, and push report changes in outputs/
  start      Run run-once every 30 minutes in the background
  stop       Stop the background auto-push loop
  status     Show whether the background auto-push loop is running
  daemon     Internal command used by start

Environment overrides:
  REPORT_AUTOPUSH_PATH=outputs
  REPORT_AUTOPUSH_INTERVAL_SECONDS=1800
  REPORT_AUTOPUSH_REMOTE=origin
  REPORT_AUTOPUSH_BRANCH=<current-branch>
  REPORT_AUTOPUSH_DIR=.git-report-autopush
USAGE
}

log() {
  printf '[report-autopush] %s\n' "$*"
}

ensure_repo() {
  git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1
}

ensure_git_identity() {
  if ! git -C "$REPO_ROOT" config user.name >/dev/null 2>&1; then
    git -C "$REPO_ROOT" config user.name "Codex Backup"
  fi
  if ! git -C "$REPO_ROOT" config user.email >/dev/null 2>&1; then
    git -C "$REPO_ROOT" config user.email "codex-backup@local"
  fi
}

pid_is_running() {
  [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" >/dev/null 2>&1
}

prepare_runtime_dir() {
  mkdir -p "$RUNTIME_DIR"
}

run_once() {
  ensure_repo || {
    log "Repository not found at $REPO_ROOT"
    return 1
  }

  ensure_git_identity
  prepare_runtime_dir

  if [ ! -e "$REPO_ROOT/$REPORTS_PATH" ]; then
    log "Report path does not exist yet: $REPORTS_PATH"
    return 0
  fi

  git -C "$REPO_ROOT" add -A -- "$REPORTS_PATH"

  if git -C "$REPO_ROOT" diff --cached --quiet -- "$REPORTS_PATH"; then
    log "No report changes to commit."
    return 0
  fi

  local message
  message="${REPORT_AUTOPUSH_MESSAGE:-reports: auto-sync $(date -u +%Y-%m-%dT%H:%M:%SZ)}"

  git -C "$REPO_ROOT" commit -m "$message" -- "$REPORTS_PATH"
  git -C "$REPO_ROOT" push "$REMOTE_NAME" "$BRANCH_NAME"
  log "Pushed report changes to $REMOTE_NAME/$BRANCH_NAME"
}

daemon() {
  prepare_runtime_dir
  echo "$$" > "$PID_FILE"
  trap 'rm -f "$PID_FILE"' EXIT

  log "Starting background loop with ${INTERVAL_SECONDS}s interval"
  while true; do
    if ! run_once; then
      log "run-once failed; will retry after ${INTERVAL_SECONDS}s"
    fi
    sleep "$INTERVAL_SECONDS"
  done
}

start() {
  prepare_runtime_dir

  if pid_is_running; then
    log "Already running with PID $(cat "$PID_FILE")"
    return 0
  fi

  rm -f "$PID_FILE"
  nohup "$SCRIPT_PATH" daemon >>"$LOG_FILE" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$PID_FILE"

  log "Started background loop with PID $pid"
  log "Log file: $LOG_FILE"
}

stop() {
  if ! pid_is_running; then
    rm -f "$PID_FILE"
    log "Auto-push loop is not running."
    return 0
  fi

  local pid
  pid="$(cat "$PID_FILE")"
  kill "$pid"
  rm -f "$PID_FILE"
  log "Stopped background loop with PID $pid"
}

status() {
  if pid_is_running; then
    log "Running with PID $(cat "$PID_FILE")"
    log "Log file: $LOG_FILE"
    return 0
  fi

  rm -f "$PID_FILE"
  log "Not running."
  return 1
}

COMMAND="${1:-}"

case "$COMMAND" in
  run-once)
    run_once
    ;;
  start)
    start
    ;;
  stop)
    stop
    ;;
  status)
    status
    ;;
  daemon)
    daemon
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
