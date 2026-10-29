# Deploying the demo on Hugging Face Spaces

> **Note (October 2026):** Hugging Face now requires a PRO subscription for Docker Spaces. For free
> hosting, follow **DEPLOY_RENDER.md** instead.

The result is a public web address such as `https://YOUR-NAME-brain-tumour-mri.hf.space` that
anyone can open. It runs on Hugging Face's free CPU machines (2 vCPU, 16 GB RAM).
One slice takes about 4–6 seconds with all three models, or about 1–2 seconds with
"Use all three trained models" switched off.

## 1. Create an account and a token (once)
1. Sign up at https://huggingface.co and confirm your email.
2. Open https://huggingface.co/settings/tokens, click **Create new token**, choose **Write**, and copy it.

## 2. Build the Space folder (on your laptop)
In the project folder, with the virtual environment active and the models and data in place
(see demo/README.md, steps 1–2):

```bash
python deploy/make_space.py
```

It creates `hf_space/` (about 150–200 MB): the app, the three models in half precision, and the
469 test-set slices for the gallery. The last line should read
`ready: .../hf_space  (... MB, 3 models, 469 gallery slices)`.

## 3. Upload it

```bash
pip install -U huggingface_hub
hf auth login                      # paste the token from step 1
hf repos create YOUR-NAME/brain-tumour-mri --type space --space-sdk docker --public --exist-ok
hf upload YOUR-NAME/brain-tumour-mri hf_space . --repo-type space
```

Replace `YOUR-NAME` with your Hugging Face username. If the upload is interrupted, run the
same `hf upload` command again; it resumes.

## 4. Wait for the build (5–10 minutes)
Open `https://huggingface.co/spaces/YOUR-NAME/brain-tumour-mri`. The status goes from
**Building** to **Running**. Click **Logs** to watch; the last lines should be:

```
[demo] 3 model(s) on cpu: runs/seed0/best.pth, runs/seed1/best.pth, runs/seed2/best.pth
[demo] test split: 469 slices from 33 patients (leakage check passed)
[demo] ready -> http://127.0.0.1:7860
```

The direct link for the presentation is `https://YOUR-NAME-brain-tumour-mri.hf.space`.

## 5. Before the presentation
- Free Spaces go to sleep after a period without visitors. Open the link 10 minutes before
  you present and analyse one slice, so it is awake and warm.
- Keep the local version (demo/README.md) ready as a backup in case the venue's internet fails.

## Updating it later
Change the code, run `python deploy/make_space.py` again, and repeat the `hf upload` command.
The Space rebuilds by itself.

## If something goes wrong
| You see | Do this |
|---|---|
| Build error mentioning `pip` | Open **Logs** and read the last 20 lines; they name the failing package. |
| `No checkpoints found` in the logs | `hf_space/runs/seed*/best.pth` were missing; rerun step 2 with the models in place. |
| `401 Unauthorized` on upload | The token is read-only or wrong; create a **Write** token and run `hf auth login` again. |
| Page shows **Sleeping** | Click **Restart this Space** and wait about a minute. |
| Want it private | In the Space's **Settings**, set visibility to Private. |
