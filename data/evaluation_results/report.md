# Evaluation report

## Test videos

| Video | Status | Loaded frames | Labeled frames | Bbox coverage | LH+ | RH+ | LF+ | RF+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IMG_0898.MOV | ok | 869 | 869 | 0.872267 | 672 | 667 | 553 | 565 |
| IMG_0912.MOV | ok | 1323 | 1323 | 0.930461 | 1128 | 1116 | 1001 | 1016 |

## Model summary

| Model | Mode | BCE loss | Exact match acc | Hamming loss | Macro F1 | Micro F1 | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| e2e | raw | 0.167747 | 0.805201 | 0.050411 | 0.968348 | 0.967892 | 83.321649 |
| masks_bbox | masked | 0.149236 | 0.816606 | 0.046875 | 0.969771 | 0.969998 | 92.772791 |
| bbox_crop | crop | 0.323722 | 0.579836 | 0.114507 | 0.927281 | 0.929026 | 92.710265 |

## Per-limb F1

| Model | LH | RH | LF | RF |
|---|---:|---:|---:|---:|
| bbox_crop | 0.959957 | 0.955448 | 0.899093 | 0.894624 |
| e2e | 0.965647 | 0.959612 | 0.969908 | 0.978227 |
| masks_bbox | 0.973648 | 0.973566 | 0.961144 | 0.970724 |

## Per-video results

| Model | Video | BCE loss | Exact match acc | Hamming loss | Macro F1 |
|---|---:|---:|---:|---:|---:|
| e2e | IMG_0898.MOV | 0.170782 | 0.812428 | 0.050921 | 0.965364 |
| e2e | IMG_0912.MOV | 0.165753 | 0.800454 | 0.050076 | 0.970045 |
| masks_bbox | IMG_0898.MOV | 0.156369 | 0.823936 | 0.045455 | 0.968414 |
| masks_bbox | IMG_0912.MOV | 0.14455 | 0.811791 | 0.047808 | 0.970554 |
| bbox_crop | IMG_0898.MOV | 0.341508 | 0.561565 | 0.122842 | 0.91677 |
| bbox_crop | IMG_0912.MOV | 0.312039 | 0.591837 | 0.109033 | 0.93354 |
