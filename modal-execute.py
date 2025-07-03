#!/usr/bin/env uvx modal run
import os
import subprocess
import time
from pathlib import Path

import modal

app = modal.App(
    image=modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .pip_install(
        "uv", "torch", "transformers", "datasets", "wandb", "albumentations", "accelerate"
    )
    .pip_install("pip==25.0.1")
    
    .add_local_file("requirements.txt", remote_path="/root/requirements.txt")
    .add_local_file("config.py", remote_path="/root/config.py")
    .add_local_file("utils.py", remote_path="/root/utils.py")
    .add_local_file("create_dataset.py", remote_path="/root/create_dataset.py")
    .add_local_file("train.py", remote_path="/root/train.py")
    
    
)

@app.function(max_containers=1, timeout=7_200, gpu="A100")
def run_py(timeout: int):
    py_process = subprocess.Popen(
        [
            "uv",
            "run",
            "/root/train.py",
            "--include_loc_tokens"
        ],
    )

    try:
        end_time = time.time() + timeout
        while time.time() < end_time:
            time.sleep(5)
        print(f"Reached end of {timeout} second timeout period. Exiting...")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        py_process.kill()

@app.local_entrypoint()
def main():
    run_py.remote(7200)