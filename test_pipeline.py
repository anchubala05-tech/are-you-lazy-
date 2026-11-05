"""
test_pipeline.py
-----------------
Lightweight sanity tests — not a full test suite, but enough to catch
"I broke the shapes" or "the scaler forgot a column" mistakes.

Run with:
    pytest tests/
"""

import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_data import _simulate, FEATURE_NAMES
from src.dataset import Scaler, ProcrastinationDataset, FEATURE_COLUMNS
from src.model import ProcrastinationNet


def test_simulate_shapes_and_ranges():
    df = _simulate(n=200, seed=0)
    assert len(df) == 200
    for col in FEATURE_NAMES:
        assert col in df.columns
    assert df["will_procrastinate"].isin([0, 1]).all()
    assert (df["procrastination_probability"] >= 0).all()
    assert (df["procrastination_probability"] <= 1).all()


def test_scaler_roundtrip(tmp_path):
    X = np.random.randn(50, len(FEATURE_COLUMNS)).astype(np.float32)
    scaler = Scaler.fit(X)
    transformed = scaler.transform(X)
    assert np.isclose(transformed.mean(axis=0), 0, atol=1e-5).all()

    path = tmp_path / "scaler.json"
    scaler.save(str(path))
    loaded = Scaler.load(str(path))
    assert np.allclose(loaded.mean, scaler.mean)
    assert np.allclose(loaded.std, scaler.std)


def test_dataset_getitem():
    X = np.random.randn(10, len(FEATURE_COLUMNS)).astype(np.float32)
    y = np.random.randint(0, 2, size=10).astype(np.float32)
    ds = ProcrastinationDataset(X, y)
    assert len(ds) == 10
    x_i, y_i = ds[0]
    assert x_i.shape == (len(FEATURE_COLUMNS),)
    assert y_i.shape == (1,)


def test_model_forward_shape():
    model = ProcrastinationNet(n_features=len(FEATURE_COLUMNS))
    batch = torch.randn(4, len(FEATURE_COLUMNS))
    out = model(batch)
    assert out.shape == (4, 1)

    probs = model.predict_proba(batch)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_model_overfits_tiny_batch():
    """Sanity check that the model + training loop can actually learn something:
    overfit a handful of points in a few steps. If this fails, something in the
    architecture or loss wiring is broken."""
    torch.manual_seed(0)
    model = ProcrastinationNet(n_features=len(FEATURE_COLUMNS))
    X = torch.randn(16, len(FEATURE_COLUMNS))
    y = (torch.rand(16, 1) > 0.5).float()

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    losses = []
    for _ in range(200):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
    assert losses[-1] < 0.1
