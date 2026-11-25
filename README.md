# Restoring Historical Photographs with Fine-Tuned Generative AI Models

**AAI-521 Final Team Project (extra-credit track) — Group 3**
Nolan Robbins (lead) · Sean McGowan · Jackson Kenyon

We build a web app that restores old, damaged photographs with four pretrained Hugging
Face models. Each model is fine-tuned on real 1930s–1970s photographs paired with
synthetically damaged copies. Selected tasks always run in this order:

```
denoise  →  inpaint  →  colorize  →  super-resolve
```

Fine-tuned models are compared against classical methods (filters, OpenCV inpainting,
bicubic upscaling), the same models before fine-tuning, and Microsoft's *Bringing Old
Photos Back to Life*.

## Status

| Component | State |
|---|---|
| Data acquisition (LoC FSA/OWI, DOCUMERICA, NVIDIA masks, DIV2K) | ✅ `scripts/download_data.py` |
| Pretrained checkpoints (7 models) | ✅ `scripts/download_models.py`, smoke-tested on GPU |
| Synthetic degradation pipeline | ✅ `restoration.data.degradations` |
| Curation + leakage-safe splits | ✅ 7,163 kept → 5,731 / 716 / 716 train/val/test (`docs/decisions.md` D7, D9) |
| Fine-tuning framework (4 tasks, resumable, Colab-ready) | ✅ `scripts/train.py`, `configs/train/*.yaml` |
| Full fine-tuning runs | ⏳ running (RTX 5060 Ti) |
| Evaluation harness (baselines vs zero-shot vs fine-tuned, bootstrap CIs) | ✅ `scripts/evaluate.py` · results ⏳ |
| BOPBtL comparison (`denoise_inpaint` task; BOPBtL in its own CPU venv) | ✅ `scripts/export_bopbtl_inputs.py`, `scripts/run_bopbtl_local.sh` (D12) · results ⏳ |
| Restoration pipeline + Flask web app (mask brush, before/after, user feedback) | ✅ `python -m restoration.webapp` |
| Submission notebooks | ✅ 01 (executed) · 02, 04 built · 03 after evaluation |

## Data

| Dataset | Role | Size | Source |
|---|---|---|---|
| LoC FSA/OWI Color Photographs, 1939–44 | train/val/test | 1,615 downloaded → 1,427 kept | [loc.gov](https://www.loc.gov/collections/fsa-owi-color-photographs/) |
| DOCUMERICA (EPA), 1972–77 | train/val/test | 6,400 sampled of 15,887 → 5,736 kept | NARA via Wikimedia Commons |
| NVIDIA Irregular Masks (Liu et al., 2018) | inpainting masks | 12,000 | [NVIDIA](https://nv-adlr.github.io/publication/partialconv-inpainting) |
| LoC FSA/OWI B&W negatives | real-damage test (no GT) | 300-image candidate pool | [loc.gov](https://www.loc.gov/collections/fsa-owi-black-and-white-negatives/) |
| DIV2K validation HR | modern-photo sanity check | 100 | [ETH Zürich](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| RealOld (Pik-Fix, WACV 2023) | real damage + expert GT | 200 | on request from the authors → `raw/realold/` |

All data lives under `$AAI521_DATA_ROOT` (never in git):

```
$AAI521_DATA_ROOT/
├── raw/<source>/{images/, manifest.csv}   # one manifest per source
├── processed/{curated.csv, splits.csv}    # curation flags + train/val/test
├── outputs/                               # smoke panels, eval results
└── logs/                                  # timestamped run logs
```

## Setup

Python 3.10+ and [uv](https://github.com/astral-sh/uv).

```bash
# 1. Environment (keep it outside OneDrive/Dropbox folders)
uv venv ~/.venvs/aai521 --python 3.11
source ~/.venvs/aai521/bin/activate        # Windows: ~/.venvs/aai521/Scripts/activate

# 2. PyTorch for your CUDA version (skip on Colab)
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
# Apple silicon: the default PyPI wheels include MPS (Metal) support
# uv pip install torch==2.11.0 torchvision==0.26.0

# 3. Project + dev tools
uv pip install -r requirements-dev.txt
uv pip install -e .
pre-commit install

# 4. Data root
cp .env.example .env    # then set AAI521_DATA_ROOT
```

**Google Colab:** mount Drive, `pip install -r requirements.txt && pip install -e .`, and
set `os.environ["AAI521_DATA_ROOT"] = "/content/drive/MyDrive/aai521_data"`.

## Notebooks (submission format)

| Notebook | Contents |
|---|---|
| `notebooks/01_data_and_eda.ipynb` | Problem definition, data acquisition, EDA, curation, splits, degradation pipeline |
| `notebooks/02_finetuning.ipynb` | Modeling methods (theory, losses, transfer learning) and fine-tuning curves |
| `notebooks/03_evaluation.ipynb` | Test-set comparison: classical vs zero-shot vs fine-tuned, CIs, qualitative results; `outputs/eval/denoise_inpaint.csv` holds the BOPBtL comparison (D12) |
| `notebooks/04_web_app.ipynb` | Integrated pipeline, Flask deployment, user testing |

Every notebook starts with the same setup cell. On **Colab** it clones the repo,
installs it, mounts Drive, and stages `MyDrive/aai521/aai521_dataset.tar` onto the
VM's local disk. Checkpoints go to `MyDrive/aai521/checkpoints`. Create the archive
once with `python scripts/pack_dataset.py` (4.7 GB) and upload it to `MyDrive/aai521/`.

## Usage

```bash
python scripts/download_data.py            # ~1 h, resumable; --sources/--limit for subsets
python scripts/download_models.py          # ~3.5 GB into the HF cache
python scripts/smoke_models.py             # zero-shot sanity check -> outputs/smoke/
python scripts/curate_and_split.py         # quality features, filters, group splits
python scripts/pack_dataset.py             # one tar for Google Drive / Colab

python scripts/train.py --config configs/train/denoise.yaml           # also: super_resolution, colorize, inpaint
python scripts/train.py --config configs/train/denoise.yaml --max-steps 150 --train-limit 64     --val-limit 16 --run-name smoke                                     # quick sanity run
python scripts/evaluate.py                 # test split -> outputs/eval/<task>.csv

# BOPBtL comparison (denoise + inpaint only; BOPBtL runs in its own venv, see D12)
python scripts/export_bopbtl_inputs.py --limit 60          # -> outputs/bopbtl/inputs/
bash scripts/run_bopbtl_local.sh                           # -> outputs/bopbtl/outputs/ (CPU)
python scripts/evaluate.py --tasks denoise_inpaint --inpaint-limit 60
python -m restoration.webapp               # http://127.0.0.1:5000
```

Or use `make data | models | smoke | curate | train | evaluate | app | test | lint | format`.

## Repository layout

```
configs/            data.yaml (sources, curation, splits), models.yaml (HF checkpoints)
docs/decisions.md   design decisions and the evidence behind them
docs/course/        assignment instructions, rubric, status update form
scripts/            CLI entry points (download, smoke test, curation)
src/restoration/
  config.py         typed (pydantic) config + data-root resolution
  data/             loc.py, commons.py, archives.py   -- dataset crawlers
                    records.py                        -- parallel, resumable downloader
                    degradations.py                   -- synthetic noise/grain/blur/scratches
                    quality.py, curation.py, splits.py -- EDA features, filters, group splits
  models/           registry.py, loaders.py, inference.py (tiled, task-level inference)
                    third_party/ddcolor/              -- vendored DDColor arch (Apache-2.0)
  training/         datasets.py (on-the-fly pairs), tasks.py (adapters), trainer.py,
                    sd_inpaint.py (LoRA)
  evaluation/       metrics.py, baselines.py, harness.py (bootstrap CIs), plotting.py
  pipeline.py       denoise -> inpaint -> colorize -> super-resolve
  webapp/           Flask app + single-page UI
  colab.py          Drive mount + dataset staging
  utils/            logging, seeding, HTTP (retries, rate limiting, atomic writes)
notebooks/          submission notebooks 01-04
tests/              unit + integration (pytest + hypothesis), no GPU needed
```

## Development standards

Black + isort (line length 100), Ruff, **strict mypy**, Google-style docstrings, pytest with
Hypothesis property tests. All of these run in pre-commit. `make lint test` must pass
before merging.

## Acknowledgements

Model weights: SCUNet/DRUNet (deepinv), Swin2SR (Conde et al.), DDColor (Kang et al.,
Apache-2.0, architecture vendored in `third_party/`), Stable Diffusion inpainting
(CreativeML OpenRAIL-M), Big-LaMa (Suvorov et al.). Photographs: Library of Congress
Prints & Photographs Division (no known restrictions) and the U.S. National Archives
(public domain).
