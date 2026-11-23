# Decision Log

Key design decisions, with the evidence behind them. Newest last.

## D1 — Pretrained model per task (2026-11-23)

| Task | Primary model | Comparison | Why |
|---|---|---|---|
| Denoise | **SCUNet** real-PSNR (`deepinv/scunet`) | DRUNet (`deepinv/drunet`) | SCUNet is *blind* (no noise-level input), which suits film grain of unknown strength in the web app. DRUNet needs σ but is the strongest Gaussian reference. ~9M params, so fine-tuning fits on a 16 GB GPU. |
| Inpaint | **Stable Diffusion 1.5 Inpainting** + LoRA (`stable-diffusion-v1-5/stable-diffusion-inpainting`) | Big-LaMa (`fashn-ai/LaMa`), OpenCV Telea | SD is the generative model named in the extra-credit spec. LoRA keeps fine-tuning feasible in fp16. LaMa is the strongest non-diffusion baseline for thin damage. |
| Colorize | **DDColor** ConvNeXt-L (`piddnad/ddcolor_modelscope`) | DDColor-tiny | State-of-the-art automatic colorization (ICCV 2023). It predicts only Lab *ab*, so luminance detail is preserved. Tiny is used for fast local ablations. |
| Super-res | **Swin2SR x4 real-world** (`caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr`) | Bicubic | Native to `transformers`. It was trained on BSRGAN real-world degradations, which are closer to old scans than bicubic-only models. |

The comparison against Microsoft's *Bringing Old Photos Back to Life* (Wan et al., 2020) is
planned for evaluation; it needs its own environment and is not yet wired in.

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
