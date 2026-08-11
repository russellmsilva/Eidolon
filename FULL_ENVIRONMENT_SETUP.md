# Full Environment Setup

This covers provisioning a working environment from scratch — no GPU, no CUDA toolchain, nothing pre-installed. If you already have a working CUDA GPU with `llama-cpp-python` built against it, you don't need this: see the Quickstart in the main README instead.

This is written against JarvisLabs (the cloud GPU provider this project was developed and tested on), but the underlying steps apply to most Linux cloud GPU instances. Budget real time for this — it is not a five-minute setup, and one step below (CUDA build detection) has a known silent-failure mode.

## 1. Choose and provision a GPU instance

This project is developed and tested against an **RTX PRO 6000 (Blackwell), 96GB VRAM**. Qwen3-Coder-Next-UD-Q4_K_XL alone uses ~49GB of that at load time, so don't go materially smaller — a 48GB card is close to the floor, not a comfortable margin.

On JarvisLabs, the default VPC option is fine (there's no meaningful choice to make there). Once the instance is up, confirm the GPU is actually visible before installing anything GPU-dependent:

```bash
nvidia-smi
```

## 2. Set up a persistent Python environment

**Important:** anything installed outside `/home` — including `apt`-installed system packages — gets wiped every time the instance is paused and resumed. Installing everything under `/home` via Miniconda sidesteps this:

```bash
cd /home/cloud
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/cloud/miniconda3
source /home/cloud/miniconda3/bin/activate
```

Create the environment:

```bash
conda create -n nosumina python=3.12 -y
conda activate nosumina
```

Add both the base and nosumina environments to the bashrc file in this order so you don't have to reactivate them everytime you ssh into the server:

```bash
echo 'source /home/cloud/miniconda3/bin/activate' >> ~/.bashrc
echo 'conda activate nosumina' >> ~/.bashrc
```

Get build tools through conda rather than `apt`, so they also survive pause/resume:

```bash
conda install -c conda-forge cmake gcc gxx -y
```

## 3. Install `llama-cpp-python` with CUDA support

Before installing, check whether the CUDA _toolkit_ (`nvcc`) is actually present — having an NVIDIA driver (`nvidia-smi` working) is not the same thing:

```bash
nvcc --version
which nvcc
```

If that fails, install a matching toolkit version into your conda env:

```bash
nvidia-smi   # note the "CUDA Version: X.Y" in the top-right corner
conda install -c nvidia cuda-toolkit=12.4 -y   # match to what you saw above
which nvcc   # should now resolve
```

Now install `llama-cpp-python`:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

**Verify the build actually detected CUDA.** After installing, confirm with:

```bash
python3 -c "from llama_cpp import llama_cpp as lib; print(lib.llama_supports_gpu_offload())"
```

This must print `True`. If it prints `True` you are done with this step and can move on to the next one.

**If it doesn't** — this is the one step in this whole setup with a known silent-failure mode: the `pip install` above can report success even when CMake couldn't find `nvcc` at build time and quietly fell back to a CPU-only build. If your check above returned `False`, rebuild with verbose logging so you can actually see what CMake detected, rather than trusting the exit code:

```bash
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir --force-reinstall --verbose 2>&1 | tee build_log.txt
grep -i "cuda" build_log.txt | head -20
```

You want to see CMake reporting `Found CUDAToolkit` or similar in that output. If it's silent on CUDA entirely, `nvcc` likely wasn't on `PATH` during the build — double check the toolkit install step above, open a fresh shell (so `PATH` picks up the conda env), and retry the standard install:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

## 4. Download the model weights

Qwen3-Coder-Next-UD-Q4_K_XL (GGUF) from Unsloth's Hugging Face repo:

**Check disk space before downloading.** Q4_K_XL quantization of an 80B-parameter MoE model is a large download — budget at least ~50GB of free disk space for the weights alone, on top of whatever your conda environment and OS are already using. Check what you actually have available first:

```bash
df -h
```

`/home/cloud/models/...` is the path used on this project's own JarvisLabs setup — adjust it to wherever you want the weights on your machine.

```bash
pip install huggingface_hub
mkdir -p /home/cloud/models
huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF \
    --include "*UD-Q4_K_XL*" \
    --local-dir /home/cloud/models/qwen3-coder-next
```

A download that runs out of disk space partway through leaves a partial, unusable GGUF file behind rather than failing cleanly up front — worth catching this before you start rather than discovering it 40GB in.

**Do a first load as a smoke test**, before relying on it inside `run-chunked`. In one terminal:

```bash
python3 -c "
from llama_cpp import Llama
llm = Llama(
    model_path='/home/cloud/models/qwen3-coder-next/<the .gguf file>.gguf',
    n_gpu_layers=-1,
    n_ctx=32768,
    verbose=True
)
"
```

(`n_gpu_layers=-1` offloads every layer to the GPU — this is the same flag `run-chunked` uses via `--n-gpu-layers -1`.)

In a second terminal, watch VRAM while that's loading to confirm the weights actually landed on the GPU rather than system RAM:

```bash
watch -n 0.5 nvidia-smi
```

You want to see usage climb toward ~49GB. (A snapshot taken mid-load can misleadingly read as near-zero VRAM usage — this is a timing artifact, not a sign of failure. Give it a few seconds.) You can also confirm from the load log itself: look for a line like `load_tensors: offloaded X/Y layers to GPU` in the `verbose=True` output above — if that count is far below the total, offload silently didn't happen even though `n_gpu_layers=-1` was set, and it's worth re-checking the CUDA build steps above.

## 5. Install and configure the bubblewrap sandbox

`bubblewrap` sandboxes every candidate `GameModel`'s execution during `backtest`/`run-chunked`, regardless of which inference backend you use. On a fresh Ubuntu 24.04 host, this needs three separate fixes beyond a plain install.

**Install (via conda, so it survives pause/resume):**

```bash
conda install -c conda-forge bubblewrap -y
```

**Fix 1 — AppArmor must explicitly permit `bwrap` to create user namespaces.** Ubuntu 23.10+ blocks this by default:

```bash
sudo tee /etc/apparmor.d/bwrap << 'EOF'
abi <abi/4.0>,
include <tunables/global>

profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,
  include if exists <local/bwrap>
}
EOF
sudo systemctl reload apparmor || sudo apparmor_parser -r /etc/apparmor.d/bwrap
```

**Fix 2 — disable the nested unprivileged-userns child-transition restriction:**

```bash
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
```

**Fix 3 — verify with an actual sandboxed command, using absolute paths:**

```bash
bwrap \
  --unshare-all --unshare-net --die-with-parent \
  --ro-bind /usr /usr --ro-bind /bin /bin \
  --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
  --proc /proc --dev /dev \
  /bin/echo "sandbox works"
```

You should see `sandbox works` printed. If you get `setting up uid map: Permission denied`, one of the two fixes above didn't take — re-check both.

## 6. Firewall lockdown (recommended)

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw --force enable
```

## 7. Everything above that doesn't survive pause/resume

`/etc/apparmor.d/bwrap`, the sysctl flag, and the UFW rules all live **outside `/home`**, so a `jl pause` / resume cycle wipes all three — even though your conda environment and code are untouched. Save this script and re-run it at the start of every session after a resume (also contains a disk usage check in case you are running this harness a lot and logs start to take up disk storage):

```bash
#!/bin/bash
# setup_sandbox.sh — run this once after every pause/resume.
# Everything here lives outside /home and gets wiped; your conda env
# and code under /home do not need this.
set -e

echo "== Firewall =="
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw --force enable

echo "== Bubblewrap availability =="
if ! command -v bwrap &> /dev/null; then
    echo "bwrap not found — reinstalling via conda-forge..."
    conda install -c conda-forge bubblewrap -y
fi

echo "== AppArmor: allow bwrap to create user namespaces =="
sudo tee /etc/apparmor.d/bwrap > /dev/null << 'EOF'
abi <abi/4.0>,
include <tunables/global>

profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,
  include if exists <local/bwrap>
}
EOF
sudo systemctl reload apparmor || sudo apparmor_parser -r /etc/apparmor.d/bwrap

echo "== AppArmor: disable the unprivileged_userns child-transition restriction =="
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0

echo "== Verifying sandbox =="
bwrap \
  --unshare-all --unshare-net --die-with-parent \
  --ro-bind /usr /usr --ro-bind /bin /bin \
  --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
  --proc /proc --dev /dev \
  /bin/echo "sandbox works"

echo "== Disk usage =="
df -h

echo "== Setup complete =="
```

```bash
chmod +x setup_sandbox.sh
./setup_sandbox.sh
```

## 8. One more pause/resume gotcha

Resuming a paused instance can assign different physical hardware, which changes its SSH host key — you'll see a scary-looking host key mismatch warning on your next SSH attempt. This is expected on this platform, not a security incident. Fix with:

```bash
ssh-keygen -R <old-host-or-ip>
```

## 9. You're set up

At this point you should be able to follow the Quickstart in the main README directly. If `llama_supports_gpu_offload()` returns `True`, `bwrap "sandbox works"` prints successfully, and your GGUF is downloaded — every prerequisite is satisfied.
