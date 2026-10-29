# Running v6 on Kaggle, step by step

Total time: about 2–3 hours on **GPU T4 x2** (two seeds train in parallel, then the third).

## Step 1. Upload the project
1. kaggle.com → **Datasets → New Dataset**.
2. Drag in `brain_tumor_mtl_v6.zip`, title it `bt-mtl-v6`, **Create**. Kaggle unzips it.

## Step 2. Create the notebook
1. **Code → New Notebook** (or **File → Import Notebook** and choose `notebooks/kaggle_run.ipynb`).
2. Right panel → **Session options**:
   - Accelerator: **GPU T4 x2**
   - Internet: **On** (needed for `pip` and the ImageNet weights; requires a phone-verified account)
3. **Add Input** → add **bt-mtl-v6** and your data dataset **bt-data**.

## Step 3. Cells

**Cell 1 – copy code and install**
```python
import glob, shutil, os
src = os.path.dirname(glob.glob('/kaggle/input/**/brain_tumor_mtl/train.py', recursive=True)[0])
shutil.copytree(src, '/kaggle/working/brain_tumor_mtl', dirs_exist_ok=True)
%cd /kaggle/working/brain_tumor_mtl
!pip install -q segmentation-models-pytorch==0.5.0 albumentations==2.0.8
!nvidia-smi -L
```

**Cell 2 – 1-epoch check (about 2 min)**
```python
!python train.py --epochs 1 --out /tmp/check
```
You must see `(matches split file)` on all three split lines, `leakage check: PASS`,
`resnet34 (imagenet)` and one epoch line. If the data folder is not found, pass it explicitly:
`--data_dir /kaggle/input/<...>/data/raw` (find it with `!find /kaggle/input -name labels.csv`).

**Cell 3 – full pipeline**
```python
!bash run_all.sh
```
This trains seeds 0 and 1 in parallel (one per GPU), then seed 2, evaluates, compares with
MAG-Net and zips the outputs. Follow progress with `!tail -n 3 logs/seed0.log logs/seed1.log`
in another cell if you run interactively.

Run it with **Save Version → Save & Run All (Commit)** so it continues after you close the tab.

## Step 4. Download
From the notebook's **Output** tab:
- `outputs_v6.zip` (results, figures, comparison, logs, training histories)
- `runs/seed0/best.pth`, `runs/seed1/best.pth`, `runs/seed2/best.pth` (≈100 MB each)

## Step 5. Read the results
`logs/evaluate.log` prints a table of every configuration on **val** and **test** and then
`selected on VALIDATION: <name>`. Report **that** configuration's test row. Then
`results/comparison/comparison.md` gives, for MAG-Net and for v5:

| column | meaning |
|---|---|
| Difference | new − baseline on the same 469 slices |
| 95% CI | patient-cluster bootstrap; if it excludes 0 the difference is real |
| p (Holm) | corrected for testing six metrics |

## Optional (decide on validation only)
- Larger encoder: `TRAIN_ARGS="--encoder efficientnet-b3" bash run_all.sh`
  (writes to the same `runs/`; use a copy of the folder if you want to keep both).
- Ablation for the report: `python train.py --no_seg_guided --seed 0 --out runs_ablation/seed0`
  shows what the segmentation-guided head contributes.

## If something fails
| Message | Fix |
|---|---|
| `Could not download the imagenet weights` | Internet is off. Turn it on in Session options. |
| `MISMATCH: split file says ...` | Wrong data folder. Use the bt-data dataset the split was made from. |
| `CUDA out of memory` | add `--batch_size 8` via `TRAIN_ARGS="--batch_size 8"` |
| Session stopped after 12 h | Should not happen (≈3 h). Each seed saves `best.pth` as it goes. |

## Explanation study (explain.py) on an existing run - no retraining
1. Upload the new zip as a **New Version** of the `bt-mtl-v6` dataset.
2. New notebook, GPU T4 (x1 is enough), Internet On. **Add Input**: `bt-mtl-v6`, `bt-data`, and
   your previous notebook (Add Input → Notebooks → Your Work) so its output (`runs/`, `results/`) is mounted.
3. Cells:
```python
import glob, shutil, os
src = os.path.dirname(glob.glob('/kaggle/input/**/brain_tumor_mtl/explain.py', recursive=True)[0])
shutil.copytree(src, '/kaggle/working/brain_tumor_mtl', dirs_exist_ok=True)
prev = os.path.dirname(os.path.dirname(os.path.dirname(glob.glob('/kaggle/input/**/runs/seed0/best.pth', recursive=True)[0])))
for d in ['runs', 'results']:
    shutil.copytree(f'{prev}/{d}', f'/kaggle/working/brain_tumor_mtl/{d}', dirs_exist_ok=True)
%cd /kaggle/working/brain_tumor_mtl
!pip install -q segmentation-models-pytorch==0.5.0 albumentations==2.0.8
```
```python
!python explain.py --ckpts runs/seed0/best.pth runs/seed1/best.pth runs/seed2/best.pth
!python compare.py --new results_xai/eval_per_slice.csv --out results_xai/comparison --new_cam "val-selected CAM"
!zip -qr /kaggle/working/xai_outputs.zip results results_xai
```
