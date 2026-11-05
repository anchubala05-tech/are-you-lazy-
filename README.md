# 💤 AI Procrastination Predictor

A small, complete deep learning pipeline that predicts the probability you're
about to procrastinate on a task, from 10 features like hours until deadline,
open browser tabs, sleep, mood, and historical completion rate.

It's a toy project (there's no real "procrastination dataset" in the wild),
but the pipeline around it is a genuine, from-scratch, production-shaped
mini-project: synthetic data generation → PyTorch `Dataset` → model →
training loop with early stopping → evaluation with metrics and plots →
CLI inference → a Streamlit demo → tests. Good as a learning reference or a
weekend-project skeleton to gut and refill with real data.

```
procrastination_predictor/
├── README.md
├── requirements.txt
├── data/
│   └── generate_data.py      # synthetic dataset generator
├── src/
│   ├── dataset.py            # PyTorch Dataset + feature scaler
│   ├── model.py              # ProcrastinationNet (feed-forward net)
│   ├── train.py              # training loop, early stopping, checkpoints
│   ├── evaluate.py           # test-set metrics + ROC/training-curve plots
│   └── predict.py            # CLI inference (flags or --interactive)
├── app.py                    # optional Streamlit demo UI
├── tests/
│   └── test_pipeline.py      # sanity tests (pytest)
└── saved_models/             # created by train.py (checkpoint, scaler, metrics)
```

## Quickstart

```bash
# 1. Set up
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# 2. Generate the synthetic dataset
python data/generate_data.py --n 5000 --out data/procrastination_data.csv

# 3. Train
python src/train.py --data data/procrastination_data.csv --epochs 100

# 4. Evaluate on the held-out test split
python src/evaluate.py --data data/procrastination_data.csv

# 5. Predict on a single task
python src/predict.py \
    --hours_until_deadline 36 --task_difficulty 7 --hours_slept 5.5 \
    --cups_of_coffee 2 --open_browser_tabs 23 --social_media_checks_per_hour 6 \
    --mood_score 4 --previous_completion_rate 0.7 --is_weekend 0 --noise_level 6

# ...or answer prompts interactively
python src/predict.py --interactive

# 6. (optional) Launch the demo UI
streamlit run app.py

# 7. (optional) Run tests
pytest tests/
```

## The features

| Feature | Meaning | Range |
|---|---|---|
| `hours_until_deadline` | Time left on the task | 0.5–168 |
| `task_difficulty` | Self-rated difficulty | 1–10 |
| `hours_slept` | Sleep the night before | 2–10 |
| `cups_of_coffee` | Coffee consumed today | 0–8 |
| `open_browser_tabs` | Tabs open right now | 1–60 |
| `social_media_checks_per_hour` | Self-reported checks/hour | 0–30 |
| `mood_score` | Self-rated mood | 1–10 |
| `previous_completion_rate` | Historical on-time completion | 0–1 |
| `is_weekend` | Weekend flag | 0/1 |
| `noise_level` | Environment distraction | 1–10 |

Target: `will_procrastinate` (binary), simulated from a weighted,
standardized combination of the features above, passed through a sigmoid,
with Gaussian noise added before thresholding — so the ground truth is
learnable but not trivially linear, and a non-linear model should out-perform
a bare logistic regression on it.

## Model

`ProcrastinationNet` is a small feed-forward network:
`10 → 32 → 16 → 8 → 1` with ReLU activations and dropout (0.2) between
hidden layers, trained with `BCEWithLogitsLoss`, Adam, weight decay, and a
`ReduceLROnPlateau` scheduler. Early stopping on validation loss (default
patience: 10 epochs). Nothing here needs a GPU — CPU training on 5,000 rows
finishes in well under a minute.

## Extending this for real

If you wanted to turn this into something real rather than a synthetic toy:

- Swap `data/generate_data.py` for a real logging pipeline (e.g. a browser
  extension or habit tracker exporting the same 10 columns, or your own
  feature set).
- Add calibration (e.g. temperature scaling) before trusting the raw
  probabilities for anything high-stakes.
- Track experiments properly (MLflow / Weights & Biases) instead of the
  flat `history.json` / `test_results.json` files here.
- Add k-fold cross-validation if your real dataset is small — 5,000
  synthetic rows is generous; real behavioral data rarely is.

## Disclaimer

This is a for-fun project with simulated data and a made-up ground-truth
function. It does not diagnose, measure, or treat anything. If you're
reading this instead of starting the thing you're procrastinating on — that
tracks.
