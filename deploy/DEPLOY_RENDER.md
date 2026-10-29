# Deploying the demo on Render (free, Docker)

Result: a public link such as `https://brain-tumour-mri.onrender.com`.

**What the free plan can run.** Render's free plan has 512 MB of memory and a shared CPU. The full
app (three models plus the model library) needs about 1–1.5 GB, so the web version is slimmed down:

- one trained model (seed 0) instead of the three-model ensemble
- exported to plain PyTorch files, so the heavy model library is not loaded
- run in reduced precision (bfloat16) to halve memory; the small classifier head stays in full
  precision. In testing, the class matched the full-precision model on every slice and masks
  agreed at Dice ≈ 0.999.

Expect 10–40 seconds per slice on the shared CPU, and about 1 minute to wake up after 15 minutes
without visitors. For the live presentation, use the laptop version (demo/README.md) or the
Cloudflare link; use the Render link to share the project afterwards.

## 1. Build the folder (on your laptop)

```bash
cd ~/Desktop/brain_tumor_mtl
source .venv/bin/activate
python deploy/make_render.py
```

Check the output for:

- `export check: max difference mask 0.00e+00, class scores 0.00e+00`
- `gallery: 469 test slices`
- `ready: .../render_app (... MB; largest file layer4.pt 50 MB, GitHub limit 100 MB)`

## 2. Put it on GitHub

One-time setup:

```bash
brew install gh
gh auth login              # choose GitHub.com, HTTPS, log in with the browser
```

Then create the repository and push:

```bash
cd render_app
git init -b main
git add .
git commit -m "Brain tumour MRI analyser - web version"
gh repo create brain-tumour-mri --public --source=. --push
```

A warning that `lite/layer4.pt` is larger than 50 MB is normal; GitHub accepts files up to 100 MB.

## 3. Create the Render service

1. Go to https://render.com and sign up **with GitHub**.
2. Click **New → Web Service**, then pick the `brain-tumour-mri` repository.
3. Render detects the Dockerfile. Set:
   - **Name:** `brain-tumour-mri`
   - **Region:** Singapore (closest to India)
   - **Instance type:** **Free**
4. Click **Deploy Web Service**.

The first build takes 5–10 minutes. When it finishes, the log shows:

```
[demo] lightweight model from lite on cpu (bf16)
[demo] test split: 469 slices from 33 patients (leakage check passed)
[demo] ready -> http://127.0.0.1:10000
```

Your link is at the top of the page, e.g. `https://brain-tumour-mri.onrender.com`.

## 4. Before showing it
Open the link a few minutes early and analyse one slice. The free service sleeps after
15 minutes without visitors and takes about a minute to wake.

## Updating it later

```bash
python deploy/make_render.py
cd render_app
git init -b main 2>/dev/null; git add . && git commit -m "Update" && git push
```

Render rebuilds automatically after each push.

## If something goes wrong

| You see in the Render log | Do this |
|---|---|
| `Out of memory (used over 512Mi)` | The free plan is too small for this run. Options: the Standard plan (2 GB), or use the laptop version with the Cloudflare link. |
| `No checkpoints found` | `render_app/lite/` is missing files; rerun step 1. |
| Build fails at `pip install torch==...` | Edit the `torch==` version in `render_app/Dockerfile` to one listed at https://download.pytorch.org/whl/cpu/torch/ and push again. |
| Page shows "Service waking up" | Normal after 15 idle minutes; wait about a minute. |

Data: Figshare brain tumour dataset (J. Cheng, 2017), CC BY 4.0. The web page states that the app
is a research prototype and not for clinical use.
