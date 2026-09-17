# Ubuntu env setup: from a fresh machine to `lerobot-calibrate`

Written 2026-09-15 for the RTX 5080 laptop. Linked from [08](../08_data_quality_research.md) week 1.
About 45–60 min, mostly downloads.

## What the machine already has (checked 2026-09-15)

| Item | State | Action |
|---|---|---|
| OS | Ubuntu **24.04.5**, kernel 7.0 | none (plan said 22.04; LeRobot is fine on 24.04. Re-check Isaac Sim / openpi support in week 2) |
| GPU | RTX 5080 Laptop, 16 GB | none |
| NVIDIA driver | **595.84, open kernel modules**, loaded with Secure Boot on (CUDA 13.2) | none: beats the ≥ 570.144 requirement |
| RAM / disk | 30 GB / 344 GB free on `/` | fine for now; datasets + checkpoints will need an external backup |
| Python | system 3.12.3 | don't use it; conda env below |
| VS Code | snap | add extensions (step 6) |
| git, gcc/make, conda, ffmpeg | **missing** | steps 1–3 |
| Serial access | user not in `dialout` | step 1 |
| GPU power limit | **45 W now** (balanced profile), 175 W max | irrelevant for calibration; set **Performance** + plug in before any training |

## 1. System packages and serial permission (needs sudo)

```bash
sudo apt update
sudo apt install -y git git-lfs build-essential curl wget v4l-utils
sudo usermod -aG dialout $USER
```

Log out and back in (or reboot) so `dialout` takes effect. Check with `groups`, which should list `dialout`.

## 2. Miniforge (conda)

```bash
cd ~/Downloads
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh      # accept defaults, answer "yes" to init
exec bash
conda config --set auto_activate_base false
```

## 3. The `lerobot` env

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install -y "ffmpeg=8" -c conda-forge   # pin 8: torchcodec 0.11 loads FFmpeg 4–8 only
ffmpeg -encoders 2>/dev/null | grep svtav1    # should print libsvtav1
```

An unpinned install pulled FFmpeg **9.0.1** on 2026-09-15, which breaks torchcodec (`Could not load
libtorchcodec`, `libavutil.so.60` not found). If `libsvtav1` is missing, use `ffmpeg=7.1.1`.

## 4. LeRobot from source, pinned to a release

Pin a tag rather than `main` (reproducibility, [09](../09_home_deployment_and_hardware.md) Part C #9).
Latest release on 2026-09-15: **v0.6.1**. On Windows you ran 0.5.2, so CLI flags may have changed; trust
`--help` over the old docs.

```bash
mkdir -p ~/code && cd ~/code
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout -b local v0.6.1
pip install -e ".[core_scripts,feetech,training,smolvla]"
pip install ipykernel                          # for measure_data_quality.ipynb in VS Code
git rev-parse HEAD                             # record this commit in your notes
```

commit: 7e241bd630a3719a56157a497ce5d08f244784f1

- **Extras:** `core_scripts` = record/replay/calibrate, `feetech` = SO-101 servos, `training` +
  `smolvla` = Part A policies. Add `pi` (π0.5) and `diffusion` later when needed.
- **PyTorch:** pip pulls the default PyPI wheel (CUDA 13.0 build). It needs driver ≥ 580 and supports
  Blackwell, so no special index is needed.

## 5. Verify

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); x=torch.randn(4096,4096,device='cuda'); print((x@x).sum().item())"
lerobot-info
lerobot-calibrate --help | head -30
python -c "import scservo_sdk; print('feetech ok')"
python -c "import torchcodec; print('torchcodec', torchcodec.__version__)"
pip check
```

Pass: `True`, `RTX 5080`, capability `(12, 0)`, a number from the matmul and no `no kernel image` error.

## 6. VS Code

```bash
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code ~/Documents/bingjian/robot_learning
```

Ctrl+Shift+P → *Python: Select Interpreter* → `~/miniforge3/envs/lerobot/bin/python`. New integrated
terminals then auto-activate the env.

Done 2026-09-15: the interpreter is pinned in `robot_learning/.vscode/settings.json`, and a Jupyter
kernel **"Python (lerobot 0.6.1)"** is registered
(`python -m ipykernel install --user --name lerobot --display-name "Python (lerobot 0.6.1)"`).
Pick that kernel for `measure_data_quality.ipynb`.

## 7. Accounts

```bash
git config --global user.name "<your name>"
git config --global user.email "<your email>"
hf auth login          # paste a write token from huggingface.co/settings/tokens
wandb login            # optional, only for training
```

## 8. Plug in the arms and find the ports

Plug in the follower and leader (one at a time the first time).

```bash
lerobot-find-port
ls -l /dev/serial/by-id/
```

**Found 2026-09-16: follower `/dev/ttyACM0`, leader `/dev/ttyACM1`.** The docs use these.

- ACM numbers follow plug order, so **plug in the follower before the leader**. If they swap, use the
  `/dev/serial/by-id/...` paths, which are tied to each board's serial number.
- **Port disappears a second after plugging in:** check `sudo dmesg | tail`. If `brltty` claimed it, run
  `sudo apt remove brltty`.
- **First commands after plugging in time out:** ModemManager is probing the port. Run
  `sudo systemctl disable --now ModemManager` (this laptop has no modem).

## 9. Ready to calibrate

Keep the same IDs as on Windows (`my_follower`, `my_leader`) so they match the dataset metadata:

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=my_leader
```

Files land in `~/.cache/huggingface/lerobot/calibration/`. **Copy them into the project with the date**
after calibrating. It's your first calibration fingerprint for drift tracking (09 Part C #4), and the
Windows calibration is not comparable because it came from a different session.

## 10. After calibration: cameras (before teleop)

The built-in HP webcam takes `/dev/video0–3`, so the C920/C922 get higher numbers. Use stable paths:

```bash
lerobot-find-cameras opencv
ls -l /dev/v4l/by-id/
v4l2-ctl -d /dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_5B3ADD8F-video-index0 --list-ctrls
```

**Current procedure: Cameractrls presets.** Settings are saved as presets in the Cameractrls app
(installed from Flathub as `hu.irl.cameractrls`). The cameras forget their settings when unplugged, so
every session:

1. Plug in the cameras, then run `flatpak run hu.irl.cameractrls`.
2. Pick **HD Pro Webcam C920** (top) and load **preset 1**. Pick **C922 Pro Stream Webcam** (wrist) and
   load **preset 2**.
3. Start teleop ([01](../01_setup_robot.md) Step 6). In the rerun viewer, the gripper tip must be sharp in
   `wrist` and `top` must not be too dark. If not, adjust in Cameractrls while teleop runs (don't press
   its preview button; the camera can only stream to one program) and re-save the preset.

What the presets set (from the 06 settings):
- **Wrist C922:** autofocus off, manual focus close enough for a sharp gripper tip (0–250, higher =
  closer); manual exposure ≈ 83 (1/120 s, in 100 µs units) against motion blur; raise gain if dark.
- **Top C920:** autofocus off, focus 0; brighter via brightness/gain or manual exposure (≤ ~300 at 30 fps).
- **Both:** dynamic framerate off, so the cameras hold 30 fps in dim light.

Checks:
- **Blank preview in Cameractrls:** usually the wrong device (the HP webcam is video0–3). To test streaming
  on its own, close LeRobot and run `ffplay -f v4l2 -input_format mjpeg -video_size 640x480 -framerate 30 <path>`
  for each camera, both at once.
- **Before recording:** OpenCV can reset some settings when LeRobot opens the camera. Check with
  `v4l2-ctl -d <path> -C focus_absolute,exposure_time_absolute,gain` while LeRobot is running.
- **Fallback without the app:** `v4l2-ctl -c ...`. Turn the auto modes off in one command
  (`focus_automatic_continuous=0 auto_exposure=1`) and set manual values in a second; in one combined
  command the manual values fail with "Permission denied".

Then the gate from 08 week 1: `lerobot-teleoperate` works with both cameras.
