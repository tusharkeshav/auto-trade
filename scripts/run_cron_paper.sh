#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
#  scripts/run_cron_paper.sh
#  Wakeup-Resilient AI Meta-Orchestrator Paper Trading Runner.
# ─────────────────────────────────────────────────────────────────

set -e

PROJECT_DIR="/home/akhil/PycharmProjects/automate-trading"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"

echo "=================================================================" >> "$LOG_DIR/cron_paper.log"
echo "⏰ Cron Triggered: $(date)" >> "$LOG_DIR/cron_paper.log"

# Execute live paper cycle with fast 0.01s idempotency guard
"$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/run_live_paper_orchestrator.py" --cron >> "$LOG_DIR/cron_paper.log" 2>&1

echo "✅ Completed: $(date)" >> "$LOG_DIR/cron_paper.log"
