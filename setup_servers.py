from pathlib import Path
import libtmux
import os
import sys

from libtmux.constants import PaneDirection

SESSION_NAME = "Model_Mornitoring"
PROJECT_DIR = Path(__file__).resolve().parent

TMUX_TTS = [
    f"cd {PROJECT_DIR}/TTS/GPT-SoVITS",
    "export CUDA_VISIBLE_DEVICES=1",
    "conda activate GPTSoVits",
    "taskset -c 14,15 python api_v2.py -p 9880",
]

# Not yet
TMUX_STT = [
        f"cd {PROJECT_DIR}/STT", 
        "export CUDA_VISIBLE_DEVICES=1",
        "source ./.venv/bin/activate",
        "taskset -c 20,21 uvicorn STT_api_calls:app --host 0.0.0.0"
        ]

TMUX_VL = [
    f"cd {PROJECT_DIR}/models/llama.cpp/build", 
    "export CUDA_VISIBLE_DEVICES=0",
    "taskset -c 8-19 ./bin/llama-server \
    -m ../../Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-Q4_K_M.gguf \
    --n-gpu-layers 99 \
    --ctx-size 32768 \
    --host 0.0.0.0 \
    --port 28401 \
    -ctk q4_0 \
    -ctv q4_0"
]


def setup_tmux():
    server = libtmux.Server()
    if server.has_session(SESSION_NAME):
        print(f"NOTICE!!!: Session '{SESSION_NAME}' already exists.")
        return

    print(f"Initializing Reminh's nerve system(internal servers): {SESSION_NAME}...")

    session = server.new_session(session_name=SESSION_NAME)
    win = session.windows[0]
    win.rename_window("Monitoring")

    pane_tts = win.panes[0]
    pane_stt = win.split(direction=PaneDirection("RIGHT"))
    pane_vl = pane_stt.split(direction=PaneDirection("RIGHT"))

    def run_commands(pane, cmds, name):
        print(f"Starting {name}...")
        pane.send_keys("source ~/.bashrc")
        for cmd in cmds:
            pane.send_keys(cmd)

    # TTS:win
    run_commands(pane_tts, TMUX_TTS, "TTS (GPU 1)")
    # STT
    if "TMUX_STT" in globals() and TMUX_STT:
        run_commands(pane_stt, TMUX_STT, "STT (GPU 1)")
    # VL model
    if "TMUX_VL" in globals() and TMUX_VL:
        run_commands(pane_vl, TMUX_VL, "VL (GPU 0)")

    win.select_layout("tiled")

    print("\nNerve Setup Complete! Attaching...")


if __name__ == "__main__":
    setup_tmux()
