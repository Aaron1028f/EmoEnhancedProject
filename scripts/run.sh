#!/usr/bin/env bash
set -euo pipefail

SESSION="emogene"
ROOT="/home/aaron/project"

# services: livekit_agent, llm, tts_gpt, tts_index, emogene, livekit_local
SERVICES=("${@:-all}")

# commands
cmd_livekit_agent='cd server/lk_exp/agent-starter-python && uv run python src/agent_feng_vid.py dev'
cmd_llm='cd server/models/LLM/src && python roleplay_api_for_lk.py'
cmd_tts_gpt='cd server/models/TTS/GPT-SoVITS && python api_v2_lk_save_wav.py'
cmd_tts_index='cd server/models/TTS/index-tts && CUDA_VISIBLE_DEVICES=0 uv run indextts/api_indextts.py --cuda_kernel --port 40000'
cmd_emogene='cd server/models/GeneFacePlusPlus && python emogene/realtime/emogene_lk_server2.py'
cmd_livekit_local='cd server/lk_exp/server_src_bin && ./livekit-server --dev --bind 0.0.0.0'

# conda-run helpers (不需要切換 shell，也不會污染彼此環境)
run_livekit_agent='conda run -n livekit bash -lc "'"$cmd_livekit_agent"'"'
run_llm='conda run -n roleplay bash -lc "'"$cmd_llm"'"'
run_tts_gpt='conda run -n GPTSoVits bash -lc "'"$cmd_tts_gpt"'"'
run_tts_index='conda run -n indextts bash -lc "'"$cmd_tts_index"'"'
run_emogene='conda run -n geneface_py310 bash -lc "'"$cmd_emogene"'"'
run_livekit_local='bash -lc "'"$cmd_livekit_local"'"'

ensure_tmux() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "請先安裝 tmux: sudo apt-get update && sudo apt-get install -y tmux" >&2
    exit 1
  fi
}

new_or_clear_window() {
  local win="$1" cmd="$2"
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -c "$ROOT" -n "$win"
    tmux send-keys -t "$SESSION:$win" "$cmd" C-m
  else
    # 若視窗存在就清空並重新執行；不存在就新建
    if tmux list-windows -t "$SESSION" | grep -qE "^[0-9]+: $win"; then
      tmux send-keys -t "$SESSION:$win" C-c
      tmux send-keys -t "$SESSION:$win" "reset" C-m
      tmux send-keys -t "$SESSION:$win" "$cmd" C-m
    else
      tmux new-window -t "$SESSION" -n "$win" -c "$ROOT"
      tmux send-keys -t "$SESSION:$win" "$cmd" C-m
    fi
  fi
}

want() {
  local name="$1"
  if [[ "${SERVICES[*]}" == "all" ]] || [[ " ${SERVICES[*]} " == *" $name "* ]]; then
    return 0
  fi
  return 1
}

main() {
  ensure_tmux
  mkdir -p "$ROOT/logs"

  if want livekit_agent; then new_or_clear_window "livekit_agent" "$run_livekit_agent"; fi
  if want llm;           then new_or_clear_window "llm"            "$run_llm"; fi
  if want tts_gpt;       then new_or_clear_window "tts_gpt"        "$run_tts_gpt"; fi
  if want tts_index;     then new_or_clear_window "tts_index"      "$run_tts_index"; fi
  if want emogene;       then new_or_clear_window "emogene"        "$run_emogene"; fi
  if want livekit_local; then new_or_clear_window "livekit_local"  "$run_livekit_local"; fi

  echo "tmux session: $SESSION"
  echo "attach 指令: tmux attach -t $SESSION"
}

main "$@"