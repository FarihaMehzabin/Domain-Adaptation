#!/usr/bin/env bash
set -euo pipefail

HOSTNAME="github.com"
GIT_PROTOCOL="https"
SKIP_INSTALL=0
TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"

usage() {
  cat <<'USAGE'
Usage: scripts/setup_github_auth.sh [options]

Installs GitHub CLI if needed, authenticates GitHub, and configures git to use gh.

Options:
  --skip-install           Skip installing gh and only run authentication.
  --hostname HOSTNAME      GitHub hostname to authenticate against. Default: github.com
  --git-protocol PROTOCOL  Preferred Git protocol for newer gh versions. Default: https
  -h, --help               Show this help message.

Authentication modes:
  1. If GH_TOKEN or GITHUB_TOKEN is set, the script uses it non-interactively.
  2. Otherwise it launches the standard gh web login flow.

Examples:
  scripts/setup_github_auth.sh
  GH_TOKEN=... scripts/setup_github_auth.sh
  scripts/setup_github_auth.sh --skip-install
USAGE
}

log() {
  printf '[setup_github_auth] %s\n' "$*"
}

run_with_privilege() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    log "Need root or sudo to install gh on this machine."
    exit 1
  fi
}

install_gh() {
  if command -v gh >/dev/null 2>&1; then
    log "gh is already installed: $(gh --version | head -n 1)"
    return 0
  fi

  case "$(uname -s)" in
    Darwin)
      if command -v brew >/dev/null 2>&1; then
        log "Installing gh with Homebrew"
        brew install gh
      else
        log "Homebrew is required to install gh on macOS."
        exit 1
      fi
      ;;
    Linux)
      if command -v apt-get >/dev/null 2>&1; then
        log "Installing gh with apt-get"
        run_with_privilege apt-get update
        run_with_privilege apt-get install -y gh
      elif command -v dnf >/dev/null 2>&1; then
        log "Installing gh with dnf"
        run_with_privilege dnf install -y gh
      elif command -v yum >/dev/null 2>&1; then
        log "Installing gh with yum"
        run_with_privilege yum install -y gh
      elif command -v pacman >/dev/null 2>&1; then
        log "Installing gh with pacman"
        run_with_privilege pacman -Sy --noconfirm github-cli
      elif command -v zypper >/dev/null 2>&1; then
        log "Installing gh with zypper"
        run_with_privilege zypper --non-interactive install gh
      else
        log "Unsupported Linux package manager. Install gh manually from https://cli.github.com/ and rerun with --skip-install."
        exit 1
      fi
      ;;
    *)
      log "Unsupported operating system: $(uname -s)"
      exit 1
      ;;
  esac

  log "Installed gh: $(gh --version | head -n 1)"
}

gh_supports_git_protocol_flag() {
  gh auth login --help | grep -q -- '--git-protocol'
}

authenticate_gh() {
  local login_cmd

  if gh auth status --hostname "$HOSTNAME" >/dev/null 2>&1; then
    log "gh is already authenticated for $HOSTNAME"
    gh auth setup-git --hostname "$HOSTNAME" >/dev/null 2>&1 || true
    gh auth status --hostname "$HOSTNAME"
    return 0
  fi

  login_cmd=(gh auth login --hostname "$HOSTNAME")
  if gh_supports_git_protocol_flag; then
    login_cmd+=(--git-protocol "$GIT_PROTOCOL")
  else
    log "Installed gh does not support --git-protocol; using version-compatible login flow"
  fi

  if [ -n "$TOKEN" ]; then
    log "Authenticating with token from environment"
    printf '%s' "$TOKEN" | "${login_cmd[@]}" --with-token
  else
    log "Launching interactive GitHub login flow"
    "${login_cmd[@]}" --web
  fi

  gh auth setup-git --hostname "$HOSTNAME"
  gh auth status --hostname "$HOSTNAME"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --hostname)
      HOSTNAME="$2"
      shift 2
      ;;
    --git-protocol)
      GIT_PROTOCOL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      log "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [ "$SKIP_INSTALL" -eq 0 ]; then
  install_gh
fi

authenticate_gh
