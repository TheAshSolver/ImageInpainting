#!/usr/bin/env bash
# Installs the files of claude_solution.zip into the project and the research repo.
#   tools/apply_solution.sh <unzipped solution dir> <APP root (has gradlew)> <REPO root (has StableDiffusion/)>
# Every file that is overwritten is first copied to <APP>/.solution_backup_<stamp>/ (outside src/, never committed by accident),
# and a file whose current content differs from the snapshot the solution was built against is reported as DRIFT.
set -euo pipefail
SRC=${1:?solution dir}; APP=${2:?app root}; REPO=${3:?repo root}
STAMP=$(date +%s); BK="$APP/.solution_backup_$STAMP"; mkdir -p "$BK"
[ -x "$APP/gradlew" ] || { echo "[apply] $APP has no gradlew"; exit 1; }
[ -f "$SRC/BASELINE_SHA256.txt" ] || { echo "[apply] BASELINE_SHA256.txt missing in $SRC"; exit 1; }

install_tree() { # <from> <to root> <prefix used in BASELINE_SHA256.txt>
  local from=$1 to=$2 prefix=$3 n=0 drift=0
  while IFS= read -r f; do
    f=${f#./}; local dst="$to/$f" want have
    if [ -f "$dst" ]; then
      want=$(grep -F "  $prefix/$f" "$SRC/BASELINE_SHA256.txt" | awk '{print $1}' | head -1 || true)
      have=$(sha256sum "$dst" | awk '{print $1}')
      if [ -n "$want" ] && [ "$want" != "$have" ]; then echo "[apply] DRIFT  $prefix/$f differs from the snapshot the solution was built on (kept in backup)"; drift=$((drift+1)); fi
      if ! cmp -s "$from/$f" "$dst"; then mkdir -p "$BK/$prefix/$(dirname "$f")"; cp -p "$dst" "$BK/$prefix/$f"; fi
    fi
    mkdir -p "$(dirname "$dst")"; cp -p "$from/$f" "$dst"; n=$((n+1))
  done < <(cd "$from" && find . -type f | sort)
  echo "[apply] $prefix: $n files installed, $drift with drift"
}
install_tree "$SRC/APP" "$APP" APP
[ -d "$SRC/REPO" ] && install_tree "$SRC/REPO" "$REPO" REPO

# files the solution deletes: the persistent HUD ticker of the Super-Resolution screen
for gone in app/src/main/java/qidk/fix/TelemetryBarView.kt; do
  if [ -f "$APP/$gone" ]; then mkdir -p "$BK/APP/$(dirname "$gone")"; mv "$APP/$gone" "$BK/APP/$gone"; echo "[apply] removed $gone (backup kept)"; fi
done
chmod +x "$APP"/tools/*.sh 2>/dev/null || true
if ! grep -q ".solution_backup_" "$APP/.gitignore" 2>/dev/null; then echo ".solution_backup_*/" >> "$APP/.gitignore"; fi
echo "[apply] done. Backups: $BK"
