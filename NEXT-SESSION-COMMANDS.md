# Next-session cheat sheet

Six steps. Copy-paste each command. Total time ~10 min (mostly Pod boot + pip).

## 1. Deploy a fresh RunPod Pod

UI form, no command. Use these settings:

```
Template:        runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
GPU:             RTX 6000 Ada (or any 24 GB+ in US-KS-2)
Region:          US-KS-2
Network Volume:  attach replicate-local-models at /workspace
Container disk:  30 GB
HTTP port:       8000
Web Terminal:    enabled
```

Click Deploy. Wait ~30 sec.

## 2. Open the Pod's Web Terminal, run setup

In the Pod's Web Terminal:

```bash
curl -fsSL https://raw.githubusercontent.com/kpashoks/replicate-clone/main/runpod-wan-animate/setup_wan_pod.sh | bash
```

Wait ~2-3 min. Watch for `[setup] Setup complete.` at the end. If any error, paste it back to Claude.

## 3. Start the wan-animate server (still in Pod terminal)

```bash
nohup /opt/wan-animate/start.sh > /var/log/wan-animate.log 2>&1 &
tail -f /var/log/wan-animate.log
```

Look for `INFO: Uvicorn running on http://0.0.0.0:8000`. Then Ctrl+C the `tail` (server keeps running).

## 4. Verify the server in your browser

Open the Pod's Connect tab → click the **port 8000** HTTP service link. Append `/debug/info` to the URL.

Expected:

```json
{
  "torch": "2.4.0+cu124",
  "cuda_available": true,
  ...
}
```

If `torch: 2.12.0+cu130` or `cuda_available: false`, paste the JSON back to Claude.

## 5. Update local `.env` + restart uvicorn

In `C:\Users\kpash\projects\replicate-local\.env`, find this line:

```
WAN_ANIMATE_ENDPOINT=https://REPLACE_WITH_NEW_POD_ID-8000.proxy.runpod.net
```

Replace with your new Pod's URL (from step 4, without `/debug/info`):

```
WAN_ANIMATE_ENDPOINT=https://<actual-pod-id>-8000.proxy.runpod.net
```

Then in the terminal running uvicorn:

```
Ctrl+C
uvicorn main:app --reload
```

(From the `backend/` directory.)

**Confirm the new URL loaded** — startup output should include:

```
WAN_ANIMATE_ENDPOINT       = https://<actual-pod-id>-8000.proxy.runpod.net
```

If it still shows `REPLACE_WITH_NEW_POD_ID`, the `.env` didn't save or uvicorn didn't actually restart.

## 6. Submit a character-swap test

Open `http://localhost:3000` → click **Character Swap (Video)**:

- Upload source video + character reference image
- Resolution: **832x480**
- replace_flag: **true**
- Click **Run**

While that runs, in the Pod's web terminal:

```bash
tail -f /var/log/wan-animate.log
```

Watch the model load + inference progress in real time.

## If anything fails

Paste back to Claude:
- The error message from the local backend log
- The error message from the Pod log (`/var/log/wan-animate.log`)
- What the `/debug/info` JSON shows

For Wan-specific errors (after CUDA is verified working), iteration loop is:

```bash
# In the Pod terminal:
nano /opt/wan-animate/server.py   # edit
pkill -f 'uvicorn.*server:app'    # stop
nohup /opt/wan-animate/start.sh > /var/log/wan-animate.log 2>&1 &
tail -f /var/log/wan-animate.log
```

Or push a fix to GitHub from local and run:

```bash
/opt/wan-animate/update_server.sh
```

— pulls latest server.py + restarts uvicorn in one command.
