# Evaluation report

## Test videos

| Model | Mode | Video | Status | Loaded frames | Labeled frames | Bbox coverage | LH+ | RH+ | LF+ | RF+ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| e2e | raw | IMG_0903.mov | ok | 699 | 699 | 0.911302 | 533 | 504 | 556 | 498 |
| e2e | raw | IMG_0899.mov | ok | 1417 | 1417 | 0.953423 | 1214 | 1171 | 1235 | 1078 |
| masks_bbox | masked | IMG_0903.mp4 | ok | 699 | 699 | 0.911302 | 533 | 504 | 556 | 498 |
| masks_bbox | masked | IMG_0899.mp4 | ok | 1417 | 1417 | 0.953423 | 1214 | 1171 | 1235 | 1078 |
| bbox_crop | crop | IMG_0903.mov | ok | 699 | 699 | 0.911302 | 533 | 504 | 556 | 498 |
| bbox_crop | crop | IMG_0899.mov | ok | 1417 | 1417 | 0.953423 | 1214 | 1171 | 1235 | 1078 |

## Model summary

| Model | Mode | BCE loss | Exact match acc | Hamming loss | Macro F1 | Micro F1 | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| e2e | raw | 0.884002 | 0.024575 | 0.425803 | 0.645478 | 0.704154 | 41.495744 |
| masks_bbox | masked | 0.310565 | 0.628072 | 0.105978 | 0.935481 | 0.935593 | 92.431387 |
| bbox_crop | crop | 0.300769 | 0.647921 | 0.095818 | 0.9426 | 0.943164 | 91.760687 |

## Per-limb F1

| Model | LH | RH | LF | RF |
|---|---:|---:|---:|---:|
| bbox_crop | 0.950316 | 0.934155 | 0.971054 | 0.914875 |
| e2e | 0.868259 | 0.846196 | 0.205607 | 0.66185 |
| masks_bbox | 0.947774 | 0.927707 | 0.931759 | 0.934683 |

## Per-video results

| Model | Video | BCE loss | Exact match acc | Hamming loss | Macro F1 |
|---|---:|---:|---:|---:|---:|
| e2e | IMG_0899.mov | 0.965196 | 0.018349 | 0.423253 | 0.626704 |
| e2e | IMG_0903.mov | 0.719407 | 0.037196 | 0.430973 | 0.597799 |
| masks_bbox | IMG_0899.mp4 | 0.32071 | 0.622442 | 0.109915 | 0.93478 |
| masks_bbox | IMG_0903.mp4 | 0.289998 | 0.639485 | 0.097997 | 0.936543 |
| bbox_crop | IMG_0899.mov | 0.299977 | 0.664079 | 0.091567 | 0.946501 |
| bbox_crop | IMG_0903.mov | 0.302373 | 0.615165 | 0.104435 | 0.933685 |
