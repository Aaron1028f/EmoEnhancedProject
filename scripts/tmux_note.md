```bash
# set configs of tmux
code ~/.tmux.conf

# list tmux sessions
tmux ls

# attach to a tmux session
tmux attach -t <session_name or session_id>

# create a new tmux session
tmux new -s <session_name>
# or
tmux new -s <session_name> -n <window_name>

# leave tmux session
Ctrl+b d

# leave all tmux sessions
tmux kill-server

# kill a specific session
tmux kill-session -t <session_name or session_id>


```