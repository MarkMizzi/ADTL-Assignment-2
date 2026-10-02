# Claude Code on UCloud with persistent chats, memory and login.
#
# UCloud wipes your home folder (~) when a job ends, and Claude Code keeps everything
# there. This script installs Claude Code on your persistent storage (first run only)
# and points ~/.claude and ~/.claude.json at that storage, so nothing is lost.
#
# Usage, at the start of every interactive job (inside tmux):
#   source setup_claude.sh /work/<your persistent folder>
#   claude
# Must be *sourced* (not run) so that `claude` ends up on your PATH.
# Safe to run again: it only installs/creates what is missing.

_cc_setup() {
  local BASE=${1:?usage: source setup_claude.sh /work/<your persistent folder>}
  BASE=${BASE%/}
  [ -d "$BASE" ] || { echo "ERROR: $BASE does not exist (is the folder mounted in this job?)"; return 1; }

  local TOOLS=$BASE/tools/claude          # Claude Code program
  local STATE=$BASE/claude_state          # chats, memory, login   (~/.claude)
  local CONF=$BASE/claude_state.json      # settings               (~/.claude.json)

  # 1. Install once onto persistent storage
  if [ ! -x "$TOOLS/node_modules/.bin/claude" ]; then
    command -v npm >/dev/null || { echo "ERROR: npm not found in this image"; return 1; }
    echo "Installing Claude Code into $TOOLS ..."
    mkdir -p "$TOOLS"
    npm install --prefix "$TOOLS" --no-fund --no-audit @anthropic-ai/claude-code || return 1
  fi
  export PATH="$TOOLS/node_modules/.bin:$PATH"

  # 2. Persistent state folder + settings file
  mkdir -p "$STATE"
  if [ ! -e "$CONF" ]; then
    if [ -f ~/.claude.json ] && [ ! -L ~/.claude.json ]; then cp ~/.claude.json "$CONF"; else echo '{}' > "$CONF"; fi
  fi

  # 3. Point the home-folder locations at persistent storage
  #    (an existing real ~/.claude folder is kept as a backup, never deleted)
  if [ -e ~/.claude ] && [ ! -L ~/.claude ]; then mv ~/.claude ~/.claude.bak.$(date +%s); fi
  ln -sfn "$STATE" ~/.claude
  ln -sfn "$CONF" ~/.claude.json

  echo "Claude Code ready ($(claude --version 2>/dev/null)); data in $STATE"
  echo "First time only: run 'claude' and log in; the login is then remembered."
}
_cc_setup "$@"; _cc_rc=$?; unset -f _cc_setup; return $_cc_rc 2>/dev/null || exit $_cc_rc
