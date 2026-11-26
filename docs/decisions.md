# Decision Log

Key design decisions, with the evidence behind them. Newest last.

## D1 — Pretrained model per task (2026-11-23)

| Task | Primary model | Comparison | Why |
|---|---|---|---|
| Denoise | **SCUNet** real-PSNR (`deepinv/scunet`) | DRUNet (`deepinv/drunet`) | SCUNet is *blind* (no noise-level input), which suits film grain of unknown strength in the web app. DRUNet needs σ but is the strongest Gaussian reference. ~9M params, so fine-tuning fits on a 16 GB GPU. |
| Inpaint | **Stable Diffusion 1.5 Inpainting** + LoRA (`stable-diffusion-v1-5/stable-diffusion-inpainting`) | Big-LaMa (`fashn-ai/LaMa`), OpenCV Telea | SD is the generative model named in the extra-credit spec. LoRA keeps fine-tuning feasible in fp16. LaMa is the strongest non-diffusion baseline for thin damage. |
| Colorize | **DDColor** ConvNeXt-L (`piddnad/ddcolor_modelscope`) | DDColor-tiny | State-of-the-art automatic colorization (ICCV 2023). It predicts only Lab *ab*, so luminance detail is preserved. Tiny is used for fast local ablations. |
| Super-res | **Swin2SR x4 real-world** (`caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr`) | Bicubic | Native to `transformers`. It was trained on BSRGAN real-world degradations, which are closer to old scans than bicubic-only models. |

The comparison against Microsoft's *Bringing Old Photos Back to Life* (Wan et al., 2020)
runs in its own environment and is scored on a dedicated `denoise_inpaint` task (D12).

## D2 — DOCUMERICA from Wikimedia Commons instead of Flickr

The Flickr API requires a key. NARA mirrors the full DOCUMERICA series to Commons, where
32,367 files collapse to **15,887 unique NARA ids** (each photo appears as both `.jpg` and
`.tif`). We sample 6,400 photos with a fixed seed, matching the status-update plan.

## D3 — LoC images: IIIF `full/full` + local area downscale to 2048 px

* Item JSON exposes only derivatives of 1024 px or less. Masters are 40–100 MB TIFFs
  (about 75 GB for the colour collection).
* The LoC IIIF server renders a master at full resolution (e.g. 4698×3280) but
  **returns HTTP 500 for every server-side scaling request** (tested: `!w,h`, `w,`,
  `pct:`).
* The full renders are JPEG at roughly quality 50 (mean luma quantiser about 29). Area
  downscaling by about 2.3× to 2048 px averages away most 8×8 block artifacts, giving
  better ground truth than the 1024 px derivative at about 0.3–0.6 MB per image.
* The search API refuses paging beyond 100,000 results, so B&W random-page sampling is
  capped at that depth.

## D4 — Data root outside the repository

The repo lives in a synced folder (OneDrive). Datasets (several GB) and checkpoints go to
`$AAI521_DATA_ROOT`, which is a local folder on Windows and a Drive path on Colab.
Python environments are also kept outside the synced folder.

## D5 — Leakage-safe splits by group

FSA/OWI and DOCUMERICA contain bracketed or consecutive frames of the same scene. Images
are linked when their 64-bit pHash differs by ≤ 8 bits, or when they share an archive id
or a normalised caption within the same source. Whole connected components are assigned
to train/val/test (80/10/10).

## D6 — Zero-shot findings that motivate fine-tuning (smoke test, 1 image)

| Task | Model | PSNR in → out (dB) |
|---|---|---|
| Denoise | SCUNet | 34.4 → 37.7 |
| Inpaint | LaMa | 19.8 → 50.6 |
| Inpaint | SD 1.5 inpaint | 19.8 → 32.7 |
| Super-res ×4 | bicubic / Swin2SR | 29.6 / 29.3 |

* **Swin2SR is no better than bicubic zero-shot.** Its BSRGAN training degradations do
  not match film scans. This is the clearest case for fine-tuning.
* **SD inpainting redraws thin scratches as dark lines** unless the mask is dilated. The
  UNet sees the mask at 1/8 resolution, so 1–4 px scratches vanish from the latent
  mask. We dilate by 6 px at working resolution and composite only that region back.
  Fine-tuning on scratch-like masks should reduce the need for dilation.
* **Colorization PSNR is below the grey input** and **DDColor fails on context-free
  crops** (a 512 px crop of sky gave a saturated orange blob). Colorization must be
  evaluated on whole images with perceptual and colourfulness metrics, not PSNR alone.
* **FSA/OWI "clean" targets are faded Kodachrome**: low saturation, warm cast.
  Fine-tuning colorization on them teaches period-faithful but faded colour; DOCUMERICA
  (1970s colour) partly offsets this. The report should discuss this trade-off.
* Slide-mount borders are common, so curation records border boxes and drops images
  whose borders exceed 25% of a side.

## D7 — Curation and split results (2026-11-23)

Downloaded: FSA/OWI colour 1,615 of 1,623 (8 access-restricted), DOCUMERICA 6,400/6,400,
FSA/OWI B&W candidate pool 300/300, NVIDIA masks 12,000, DIV2K 100.

| Source | Downloaded | blurry | borders | near-grey | noisy | too small | **Kept** |
|---|---|---|---|---|---|---|---|
| FSA/OWI colour | 1,615 | 68 | 30 | 6 | 74 | 27 | **1,410** |
| DOCUMERICA | 6,400 | 233 | 355 | 4 | 310 | 9 | **5,489** |

Kept images form 6,045 groups. 5,513 are singletons; the largest has 12 images, all
genuine same-scene shoots (e.g. "Civil Air Patrol Base, Bar Harbor, Maine"), so the
pHash ≤ 8 + caption rule is not over-merging.

| Split | FSA/OWI colour | DOCUMERICA | Total |
|---|---|---|---|
| train | 1,146 | 4,375 | 5,521 |
| val | 132 | 557 | 689 |
| test | 132 | 557 | 689 |

## D8 — Fine-tuning design (2026-11-23)

* **One trainer, per-task adapters** (`restoration.training`). Degraded/clean pairs are
  synthesised on the fly, so every epoch sees new damage; validation pairs are seeded by
  index and fixed. The trainer validates at **step 0**, so each run records the classical
  baseline, the zero-shot model and the fine-tuned model on identical data.
* **Transfer-learning strategy.** SCUNet, Swin2SR and DDColor are fully fine-tuned at a
  low LR (1–2e-5, cosine schedule with warm-up) to adapt rather than overwrite their
  priors. SD 1.5 inpainting uses **LoRA** (r = 16 on attention projections; 3.2M of 860M
  parameters, LR 1e-4).
* **Losses.** Charbonnier for denoising; L1 for SR; L1 + 0.5·LPIPS + 0.5·colourfulness-gap
  for colorization (L1 alone regresses toward desaturated means); the standard
  ε-prediction MSE for diffusion. For diffusion, the mask is downsampled to latent
  resolution with **max-pooling** so thin scratches survive.
* **Resumable, pickle-free artefacts.** `best.safetensors` / diffusers LoRA plus
  `last_state.pt` and `metrics.csv`. A disconnected Colab session resumes by rerunning
  the same command.
* **Memory on 16 GB (bf16)**, measured peaks: SCUNet bs 4 = 7.8 GiB, Swin2SR bs 8 =
  11.5 GiB, DDColor-L bs 2 = 7.8 GiB, SD-LoRA bs 2 + gradient checkpointing ≈ 5 GiB.
  Gradient accumulation keeps the effective batch at 8 (4 for SD). The process is
  capped at 95 % of VRAM because the Windows driver otherwise silently spills into
  system RAM (≈ 10× slower) instead of raising OOM.

Smoke runs (64 train / 16 val images, 150 steps; val PSNR in dB):

| Task | Classical baseline | Zero-shot | 150 steps |
|---|---|---|---|
| Denoise (SCUNet) | 26.45 (noisy input) | 33.84 | 35.45 |
| Super-res ×4 (Swin2SR) | 31.87 (bicubic) | 31.44 | 34.01 |
| Colorize (DDColor-L, faded targets) | 19.64 (grey) | 20.85 | 25.28 (LPIPS 0.225 → 0.126) |

## D9 — EDA-driven data revisions

* **Border rule relaxed (25 % → 45 %).** The border detector also fires on dark, low-key
  scenes; dropping them biased the corpus toward daylight. Borders are cropped at load
  time anyway. This recovered 360 images: 7,163 kept → 5,731 / 716 / 716.
* **Dye fading.** FSA/OWI Kodachromes are warm and desaturated (median colourfulness ≈ 27
  vs 43 for DIV2K); many DOCUMERICA Ektachromes have magenta casts. Colorization targets
  get a tempered grey-world + L-stretch + 15 % chroma correction
  (`degradations.correct_fading`), so the model does not learn to paint fading. This is
  evaluated as an ablation.
* **The noise estimator responds to texture** (DIV2K scores the "noisiest"), so curation
  uses per-source percentiles, not absolute thresholds.

## D10 — SD inpainting composite fix

Pasting the whole generation-dilated band back damaged intact context. On 12 val images
(zero-shot), pasting only the true hole (+1 px, 2 px feather) with 3 px latent dilation
improved PSNR 28.24 → 30.76 and SSIM 0.886 → 0.915.

OpenCV Telea still scores higher on pixel metrics (PSNR 33.9, hole-PSNR 23.1 vs 20.7).
That is expected: smooth interpolation is mean-seeking, which PSNR rewards on thin
scratches. Inpainting is therefore evaluated **per hole-size bin** and with LPIPS, where
generative fill should win on large holes.

## D11 — Apple-silicon (MPS) support for local inference (2026-11-24)

`restoration.utils.device.default_device()` picks CUDA → MPS → CPU; `empty_cache()` and
`synchronize()` dispatch to `torch.cuda` or `torch.mps`. Tested on an Apple M4 (24 GB),
torch 2.11.0, macOS wheels from PyPI.

* **SD inpainting runs in fp16 on MPS.** It matches fp32 within 3/255 per pixel (no
  NaNs or black images) at half the weight memory, and runs at the same speed.
* **Big-LaMa (TorchScript) runs on MPS as is**, so it needs no CPU fallback.
* **Fine-tuning stays fp32 on MPS** (`resolve_precision`). On a 3-step check, bf16 and
  fp16 autocast trained SCUNet and Swin2SR but were no faster. DDColor fails under
  both (`div_true_strided_float_bfloat` / `_half`: MPS lacks the mixed-dtype kernel).
* **`PYTORCH_ENABLE_MPS_FALLBACK` is not needed**: all four tasks ran with no
  unsupported-op errors.

Web-app timings (512 px input, pretrained weights, warm models; first request adds model
load time):

| Task | MPS (s) | Note |
|---|---|---|
| Denoise (SCUNet) | 0.9 | |
| Inpaint (SD 1.5, 30 steps, fp16) | 42 | ≈ 1.4 s/step |
| Colorize (DDColor-L) | 0.3 | |
| Super-res ×4 (Swin2SR) → 2048 px | 35 | 9 tiles; ≈ 2× faster than CPU |

## D12 — BOPBtL baseline on a `denoise_inpaint` task (2026-11-25)

**Scope.** *Bringing Old Photos Back to Life* (Wan et al., 2020) removes grain and scratches
but neither colorizes nor upscales, so it is compared only with our denoise + inpaint
steps. None of the existing tasks fit (`denoise` has grain only, `inpaint` holes only,
`end_to_end` grey and downsampled inputs), so `evaluate.py --tasks denoise_inpaint` adds one:
test-split colour crops at **512 px** (the inpaint `patch_size`: SD 1.5's native size, the
same crops and masks as the `inpaint` task, and a multiple of 16 so BOPBtL keeps the size),
film grain (`make_denoise_pair`) and then scratches/holes (`make_inpaint_pair`) with the
same procedural/NVIDIA split by index parity. The target is the clean crop. The builder
lives in `restoration.evaluation.denoise_inpaint`, so `evaluate.py` and
`scripts/export_bopbtl_inputs.py` cannot drift.

**Where BOPBtL runs.** On a Mac, CPU, in its own venv (`scripts/run_bopbtl_local.sh`:
`~/bopbtl` at commit `33875ec`, `~/.venvs/bopbtl`, torch 2.11; none of it is in git or our
requirements). We run only its stage 1, the two commands `run.py --with_scratch` issues
(`Global/detection.py --input_size full_size`, then `Global/test.py
--Scratch_and_Quality_restore`, non-HR). Two shims, no edits to their code: `torch.load`
gets `map_location="cpu"` (their checkpoints were saved on CUDA) and `weights_only=False`
(the checkpoints predate that default). Measured: 267 s for 100 images at 512 px (≈ 2.7
s/image including model load), so Colab is unnecessary.

**Fairness choices.**

* **Auto mask vs GT mask.** BOPBtL detects its own scratches, so `ours … (auto mask)`
  (`RestorationPipeline.run(mask=None, auto_mask=True)`) is the fair comparison.
  `ours … (GT mask)` and the classical NLM → Telea row get the true mask and are upper
  bounds.
* **8-bit inputs.** Inputs are rounded to /255 in the builder, so every method sees
  exactly the bytes BOPBtL reads from the exported PNG.
* **Size.** `detection.py` resizes to a multiple of 16 and `test.py` keeps the size, so
  512 px in gives 512 px out and no resampling touches the metrics. Should a size ever
  differ, the loader resizes back (bicubic) and logs each image.
* **Faces off.** `run.py` always runs face detection and enhancement (dlib) after stage 1.
  Our crops are mostly scenes, and stage 1's output is exactly what `run.py`
  writes to `final_output` for an image without faces, so we skip dlib entirely.
* **Fine-tuned rows** appear only when both the denoise and inpaint checkpoints exist (a
  half fine-tuned row would be mislabelled).

**Zero-shot results (interim; fine-tuned rows pending).** First 100 test images, seed 521.
These rows are final; the fine-tuned rows will be added beside them. Mean ± std over
images; 2 h 24 min on MPS for the two SD rows.

| Method | PSNR | SSIM | LPIPS ↓ | hole PSNR | s/image |
|---|---|---|---|---|---|
| classical (NLM → Telea), GT mask | 26.45 ± 5.13 | 0.711 | 0.443 | 21.82 | 0.17 (MPS) |
| ours zero-shot, GT mask | 28.25 ± 5.85 | 0.786 | 0.270 | 21.70 | 42.3 (MPS) |
| ours zero-shot, auto mask | 18.53 ± 7.11 | 0.731 | 0.430 | 8.40 | 44.1 (MPS) |
| BOPBtL | 17.29 ± 5.81 | 0.658 | 0.456 | 8.91 | ≈ 2.7 (CPU, offline) |

By `mask_type` (PSNR / LPIPS ↓; 50 images each):

| Method | procedural | NVIDIA |
|---|---|---|
| classical, GT mask | 29.60 / 0.381 | 23.30 / 0.504 |
| ours, GT mask | 31.85 / 0.223 | 24.66 / 0.317 |
| ours, auto mask | 23.76 / 0.344 | 13.30 / 0.516 |
| BOPBtL | 21.75 / 0.367 | 12.83 / 0.544 |

Ours (auto mask) − BOPBtL, paired bootstrap (5,000 resamples, 95% CI): PSNR +1.24 dB
[0.66, 1.97], better on 73/100 images; SSIM +0.073 [0.064, 0.083], 93/100; LPIPS −0.025
[−0.038, −0.013], 66/100. So off the shelf, with neither trained on our data, our
pipeline slightly but significantly beats BOPBtL on every whole-image metric, while
BOPBtL is a little ahead inside the holes. The degradations are our own synthetic ones,
and real damaged prints are untested.

With auto masks, both pipelines leave most damage in place (≈ 10 dB below the GT-mask
row), so **the fair comparison mostly measures scratch detection**. On the first 8
images, BOPBtL's detector (run on a 256 px-short-side copy) found thin procedural
scratches on 3 of 4 images (recall 0.59–0.84, 0 on the fourth) against 0.27–0.34 for our
`detect_scratches`. Neither finds the large NVIDIA holes (recall ≤ 0.19; both score
≈ 13 dB there), since neither was built for blobs. Fine-tuning does not touch
`detect_scratches`, so this bottleneck should survive it. Report this task **per
`mask_type`**.

The overall comparison (fine-tuned ours vs BOPBtL, and where any margin comes from) is
left until the fine-tuned rows exist.

**Handoff.** One person (Sean, on the Mac) runs the whole chain: export, then BOPBtL, then
`evaluate.py`, all with the same `--limit`/`--inpaint-limit` and `--seed`, so the loader's
seed/limit check covers it. Nobody else needs the BOPBtL environment. The results are
copied to `MyDrive/aai521/outputs/`:

* `eval/denoise_inpaint.csv`: the per-image rows notebook 03 reads.
* `bopbtl/outputs/<index:04d>.png` and `bopbtl/manifest.csv`: BOPBtL's restored images,
  for qualitative panels. Image `i` is row `image == i` in the CSV.

Rerun `evaluate.py` and copy the CSV again once the fine-tuned checkpoints exist. BOPBtL
does not need to rerun as long as the seed and limit are unchanged.
