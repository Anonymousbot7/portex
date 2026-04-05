# portex

Domain-Adversarial Neural Network (DANN) for cross-domain prediction of environmental factors **E** from genotype matrix **Z**.

**Setup**: train on source domain (Z_source, E_source known), adapt to target domain (Z_target, E unknown), predict E on target.

---

## Installation

**From GitHub (recommended):**

```bash
pip install git+https://github.com/Anonymousbot7/portex.git
```

**From a specific branch or tag:**

```bash
pip install git+https://github.com/Anonymousbot7/portex.git@main
pip install git+https://github.com/Anonymousbot7/portex.git@v0.2.0
```

**Clone and install locally:**

```bash
git clone https://github.com/Anonymousbot7/portex.git
cd portex
pip install .
```

**Dependencies** (`torch` and `numpy`) are installed automatically.

After installation, import directly:

```python
from portex import DANN
```

---

## Quick Start

```python
from portex import DANN

model = DANN(input_dim=Z_source.shape[1])
model.fit(Z_source, E_source, Z_target)

E_target_hat = model.predict(Z_target)
```

The return shape of `predict()` automatically matches your `E_source`:

| `E_source` shape | `predict()` returns |
|---|---|
| `(N_s,)` — single dimension | `(N_target,)` |
| `(N_s, G)` — G dimensions | `(N_target, G)` |

---

## Parallel Training

For high-dimensional E, train all G dimensions in parallel with `n_jobs`:

```python
# use 4 parallel workers
model = DANN(input_dim=d, n_jobs=4)

# use all available CPUs
model = DANN(input_dim=d, n_jobs=-1)
```

`n_jobs=1` (default) runs sequentially. Each dimension is an independent process, so speedup scales roughly linearly with `min(n_jobs, G, n_cpu)`.

---

## Tuning Parameters

| Parameter | Default | Description |
|---|---|---|
| `input_dim` | *(required)* | Number of columns in Z |
| `latent_dim` | `64` | Dimension of shared feature representation |
| `hidden_dim_F` | `128` | Hidden units in feature extractor |
| `hidden_dim_DP` | `64` | Hidden units in discriminator and predictor |
| `warmup_epochs` | `50` | Epochs to pre-train F+D before adversarial phase |
| `epochs` | `800` | Adversarial training epochs |
| `lr` | `1e-3` | Learning rate |
| `weight_decay` | `0.008` | L2 regularisation (discriminator and predictor) |
| `lambda_domain` | `0.05` | Weight of domain-adversarial loss |
| `clip_grad_norm` | `5.0` | Gradient clipping max norm (`None` to disable) |
| `seed_base` | `0` | Dimension g uses seed = seed_base + g |
| `n_jobs` | `1` | Parallel workers (`-1` = all CPUs, `1` = sequential) |

Defaults match the original experimental settings exactly.

---

## Baseline NN

For comparison, `fit_base` trains the same F+P architecture on source data using MSE only — no domain adaptation. This serves as a direct baseline against DANN.

```python
model = DANN(input_dim=Z_source.shape[1])

# Train both DANN and baseline on the same object
model.fit(Z_source, E_source, Z_target)       # DANN
model.fit_base(Z_source, E_source)            # baseline (no Z_target needed)

E_dann = model.predict(Z_target)              # DANN predictions
E_base = model.predict_base(Z_target)         # baseline predictions

mse_dann = model.mse(Z_target, E_true)        # DANN MSE
mse_base = model.mse_base(Z_target, E_true)   # baseline MSE
```

`fit_base` accepts an optional `epochs` argument (defaults to `self.epochs`) and supports `n_jobs` parallel training the same way as `fit`.

---

## API Reference

```python
model = DANN(input_dim, **kwargs)

# DANN
model.fit(Z_source, E_source, Z_target, verbose=False) -> self
model.predict(Z)                                        -> (N,) or (N, G)
model.mse(Z, E_true)                                   -> float or (G,) array

# Baseline NN (MSE only, no domain adaptation)
model.fit_base(Z_source, E_source, epochs=None, verbose=False) -> self
model.predict_base(Z)                                           -> (N,) or (N, G)
model.mse_base(Z, E_true)                                      -> float or (G,) array
```

**`fit`**
- `Z_source`: `(N_s, d)` — source genotypes
- `E_source`: `(N_s,)` or `(N_s, G)` — source E values
- `Z_target`: `(N_t, d)` — target genotypes (E not used in training)

**`fit_base`**
- `Z_source`, `E_source`: same as above
- `epochs`: overrides `self.epochs` if provided; otherwise uses `self.epochs`
- No `Z_target` needed

**`predict` / `predict_base`**
- Returns `(N,)` if `E_source` was 1-D, `(N, G)` if 2-D

**`mse` / `mse_base`**
- Returns `float` if E is 1-D, `(G,)` array of per-dimension MSE if 2-D

---

## Input Validation

The following are checked at `fit()` and `predict()` time, with clear error messages:

- `Z_source` and `Z_target` must be 2-D
- `Z_source.shape[1] == Z_target.shape[1]` (same number of features)
- `Z.shape[1] == input_dim` (consistent with construction)
- `E_source.shape[0] == Z_source.shape[0]` (rows match)
- `E_source` must be 1-D or 2-D

---

## Verbose Training

```python
model.fit(Z_source, E_source, Z_target, verbose=True)
# [dim 0] epoch 100/800  task_mse=0.1234  dom_loss=1.3821  alpha=0.020
# [dim 0] epoch 200/800  task_mse=0.0987  dom_loss=1.2104  alpha=0.119
# ...
```

---

## How It Works

1. **Feature extractor F**: maps Z → shared latent representation h
2. **Predictor P**: maps h → E (trained with MSE loss on source)
3. **Domain discriminator D**: tries to distinguish source vs target from h
4. **Adversarial training**: F is trained to fool D (via gradient reversal) while minimising prediction MSE on source — forcing h to be domain-invariant

Domain label convention: source = 1, target = 0.
