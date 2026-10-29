# Training run logs – Brain tumor segmentation and classification (BCSE497J)

Raw Kaggle logs behind the "Three training runs" table in the report and paper, plus the ablation run.
Every run uses the same frozen patient-level split: 167 / 33 / 33 patients, 2,186 / 409 / 469 slices.
Leakage check: PASS in every log. The logs are unedited copies of the Kaggle output.

## Where each number in the table comes from

| Run | Setting chosen on validation | Accuracy | Dice | Gain over MAG-Net [95% CI] | McNemar | Log file (line) |
|---|---|---|---|---|---|---|
| Run 1 | 3-model ensemble + flip | 98.72% | 0.8395 | +3.84 pts [0.80, 7.73] | 22 / 4, p < 0.001 | `run1_ensemble_flip/run1_train_eval_compare.log` (selection line 77, test lines 81–83, comparison line 93) |
| Run 2 | Seed 0 + flip | 98.08% | 0.8305 | +3.20 pts [0.25, 6.61] | 21 / 6, p = 0.006 | `run2_seed0_flip/outputs/logs/evaluate.log` (lines 17–23), `run2_seed0_flip/outputs/logs/compare.log` (line 3) |
| Run 3 | Seed 0 + flip | 97.87% | 0.8296 | +2.99 pts [0.18, 6.50] | 20 / 6, p = 0.009 | `run3_seed0_flip/run3_train_eval_compare.log` (selection line 99, test lines 103–105, comparison line 115) |

Mean ± SD over the three runs: accuracy 98.22 ± 0.44%, Dice 0.833 ± 0.006.
MAG-Net on the same split (from the same comparison lines): accuracy 94.88%, Dice 0.7604.

## Folder contents

- `run1_ensemble_flip/run1_train_eval_compare.log` – training of 3 seeds, validation-based selection, test evaluation, paired comparison with MAG-Net and with the previous multi-task model.
- `run2_seed0_flip/run2_kaggle_console.log` – full Kaggle notebook console for run 2 (training, evaluation and comparison, with timestamps).
- `run2_seed0_flip/run2_explain_compare.log` – run 2 explanation step (Grad-CAM vs HiResCAM, selected on validation) and the comparison re-run with the selected heatmap method.
- `run2_seed0_flip/outputs/` – the `outputs_v6.zip` download from Kaggle for run 2:
  - `logs/seed0.log`, `seed1.log`, `seed2.log` – per-seed training logs
  - `logs/evaluate.log`, `logs/compare.log` – evaluation and comparison logs
  - `runs/seed*/history.csv`, `run_config.json`, `best_summary.json` – per-epoch history, configuration and best epoch of each seed
  - `results/eval.json`, `results/eval_per_slice.csv`, `results/comparison/` – metrics, per-slice predictions and the comparison tables
- `run3_seed0_flip/run3_train_eval_compare.log` – training, selection, test evaluation, comparison and explanation step for run 3.
- `ablation_no_mask_pooling/ablation_eval_compare.log` – evaluation and MAG-Net comparison of the ablation model trained with `--no_seg_guided` (selected on validation: seed 1 + flip; 98.29%, Dice 0.8346, McNemar 21 vs 5, p = 0.002). Copied from the Kaggle console output.

## Notes for reading the logs

- "Disagreements" in the table = slices only the proposed model got right / only MAG-Net got right (exact McNemar test).
- Each comparison block also prints a `p_holm` value (Holm correction across the six metrics compared in that block). It changes with the set of metrics tested alongside accuracy, so the same run can show different `p_holm` values in different blocks. The paper reports this in Section V.
- Model checkpoints (`best.pth`) are not included because of their size.
