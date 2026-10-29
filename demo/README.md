# Live demo app

A local web app for the presentation. Pick a test-set slice, or drop in your own MRI slice. The app shows:

- the predicted tumour mask over the slice, with the expert outline when one is known
- the tumour type with the probability of each class
- Grad-CAM and HiResCAM heatmaps for the predicted class
- the Dice score of the mask, and how much of each heatmap falls inside the tumour

The **Results** tab shows the three runs against MAG-Net. The **How it works** tab shows the pipeline and the architecture.

It runs on a laptop CPU (Mac, Windows or Linux). It needs **no internet** once installed: no CDN, no web fonts.

## 1. One-time setup (about 10 minutes)

```bash
cd brain_tumor_mtl
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r demo/requirements.txt
```

## 2. Put the trained models and the data in place

**Models.** Unzip `run1_results.zip` (from the Kaggle Output tab) inside `brain_tumor_mtl/`, so you have:

```
brain_tumor_mtl/runs/seed0/best.pth
brain_tumor_mtl/runs/seed1/best.pth
brain_tumor_mtl/runs/seed2/best.pth
brain_tumor_mtl/results/eval.json
```

**Data** (optional, but needed for the test-set gallery). On Kaggle, open your `bt-data` dataset and click **Download**. Unzip it so that `brain_tumor_mtl/data/raw/labels.csv`, `images/` and `masks/` exist. You can also pass any other location with `--data_dir`.

## 3. Start it

```bash
bash demo/run_demo.sh              # Windows: double-click demo\run_demo.bat
```

The browser opens at http://127.0.0.1:8000. The terminal should print:

```
[demo] 3 model(s) on cpu: runs/seed0/best.pth, runs/seed1/best.pth, runs/seed2/best.pth
[split] test :  33 patients,  469 slices  (matches split file)
[demo] dataset .../data/raw: 469 test slices in the gallery
[demo] ready -> http://127.0.0.1:8000
```

## Options

| Option | Use |
|---|---|
| `--ckpts a.pth b.pth` | models in another location |
| `--data_dir PATH` | dataset in another location |
| `--port 8080` | if port 8000 is busy |
| `--device mps` | Apple-silicon GPU (CPU is the safe default) |
| `--no_browser` | don't open the browser automatically |

## Before the presentation

1. Start the app 5 minutes early and analyse one slice. The first run loads everything into memory.
2. Choose three slices in advance, one per tumour type, and note their IDs.
3. If a slice feels slow, switch off **Use all three trained models** (about 3× faster). The setting reported in the report is all three models with flip averaging.
4. Turn off Wi-Fi notifications; the app needs no network.
5. Use the browser's full-screen mode (F11, or Ctrl+Cmd+F on a Mac). Click any image to enlarge it for the room.

## If something goes wrong

| Message | Fix |
|---|---|
| `No checkpoints found` | The `runs/seed*/best.pth` files are missing; see step 2. |
| `no dataset folder found` | The gallery is off. Uploads still work; add `--data_dir`. |
| `address already in use` | Use `--port 8080`. |
| `Missing packages` | Run `pip install -r demo/requirements.txt` inside the virtual environment. |
