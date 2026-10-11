# InterviewLens

Practice tool for job interviews. Record yourself answering a question and get
feedback on how you came across: where you looked, head movement, facial
expression, speaking rate, filler words and pauses. Every score comes with the
evidence behind it.

This is v2 of my IEEE 2025 paper *"A Real-Time Multi-Model System for
Personality and Interview Performance Analysis Using Transformer-Based Feature
Fusion"*. v1 was visual only, per frame, with concat + MLP fusion and a small
private validation set. v2 adds audio and text, models time, uses
cross-attention fusion, and is evaluated on public benchmarks with human
labels, a fairness audit and calibrated output.

It is meant for candidates, not employers. See [ETHICS.md](ETHICS.md).

## Status

| Milestone | What | State |
|---|---|---|
| M0 | Repo scaffold, CI, configs, download script | done |
| M1 | Face pipeline (MediaPipe crops, FPS benchmark) | next |
| M2 | Emotion (FER2013 / FER+) | - |
| M3 | Head pose (300W-LP -> AFLW2000, BIWI) | - |
| M4 | Gaze (MPIIFaceGaze) | - |
| M5 | Audio + text features | - |
| M6 | Fusion + baselines + ablations | - |
| M7 | Fairness + calibration | - |
| M8 | Real-time app | - |
| M9 | Release | - |

No results yet. Result tables will be generated from run logs, not typed by hand.

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"          # add ,track for wandb
pre-commit install
```

## Usage

```bash
python -m interviewlens.train --help            # hydra config + overrides
python -m interviewlens.train train.epochs=5 wandb.enabled=true
bash scripts/download_public.sh                 # FER2013/FER+, 300W-LP, AFLW2000, MPIIFaceGaze
```

## Checks

```bash
ruff check . && ruff format --check .
mypy
pytest
```

CI runs the same on every push.

## Layout

```
configs/              hydra configs
data/                 gitignored, only README, LICENSES.md and splits/
scripts/              dataset download / prep
src/interviewlens/    package code
tests/
```

## Data

Raw data is never committed. Dataset sources and licenses are in
[data/LICENSES.md](data/LICENSES.md).

## License

MIT
