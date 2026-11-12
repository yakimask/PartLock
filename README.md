# PartLock

**Strictly local hierarchical motion tokens for endless, switchable text-to-motion streaming.**

Paper: [`paper/main.pdf`](paper/main.pdf) (LaTeX source: [`paper/main.tex`](paper/main.tex))
Author: Kaiji Hiruma (Independent Researcher)

## Idea

Every 4 frames of motion are encoded into one **global** token (whole body) and six **part** tokens
(torso, head, right/left arm, right/left leg).
Each body part is decoded **only** from `[global token, its own part token]`:

- **Locality by construction.** Replacing one part's tokens changes that part and nothing else
  (measured change in all other parts: exactly 0).
- **Coordination through the global token.** The shared global token is the only channel between parts.
- **Strictly causal.** All convolutions are left-padded, so the tokenizer looks 0 frames into the future
  and streaming decoding equals offline decoding (max. difference 2.1e-15 in float64).

A 30.5M-parameter decoder-only Transformer generates the 7 tokens of each step from CLIP text features
added at every token position, and runs without end. When the text changes, all context except the last
step is dropped, so the motion changes action on command without returning to a rest pose.

## Files

| File | Role |
|---|---|
| `tvq272c.py` | Causal hierarchical tokenizer (`CHierTok`) and its training |
| `tvq272.py`, `tvq.py`, `vq_mix.py` | Data loading, metrics, VQ layers, non-causal reference tokenizer |
| `rf_probe.py` | Look-ahead / look-back of encoder and decoder (input gradients) |
| `stream64.py` | Streaming vs. offline decoding in float64 |
| `causal_check.py` | Whole-clip reconstruction and right-arm swap (locality) |
| `gen_prep.py` | Encode clips into tokens and captions into CLIP embeddings |
| `gen_ar.py` | Token generator (`Gen`) and its training |
| `gen_sample.py` | 10 s / 60 s generation and plausibility measures |
| `switch_test.py` | Text switching every 6 s (`--keep1step` = the switching rule) |
| `stop_test.py` | Run/walk, then "a person stops and stands up straight." |

## Data

This repository contains **no motion data and no trained weights**.

The code uses the 272-dimensional HumanML3D representation released by the MotionStreamer authors
([code](https://github.com/zju3dv/MotionStreamer),
[processing](https://github.com/Li-xingXiao/272-dim-Motion-Representation)).
That data is derived from [HumanML3D](https://github.com/EricGuo5513/HumanML3D) and
[AMASS](https://amass.is.tue.mpg.de/) and is for academic use only; read the AMASS license before use.

Expected layout (set `PARTLOCK_DIR` to the folder that contains `data/`):

```
$PARTLOCK_DIR/
  data/motion_raw/h272/
    motion_data/...        # *.npy, (T, 272), 30 fps
    texts/texts/*.txt      # HumanML3D captions
    split/{train,val,test}.txt
    mean_std/{Mean,Std}.npy
  ckpt/                    # checkpoints are written here
  runs/                    # results (json) are written here
```

## Reproduce

Tested with Python 3.13, PyTorch 2.14 (CUDA), NumPy 2.5, transformers 5.22, on one RTX 3060 (12 GB).

```bash
export PARTLOCK_DIR=/path/to/workdir      # Windows: set PARTLOCK_DIR=C:\path\to\workdir

python tvq272c.py                          # train the causal tokenizer (6000 steps, ~40 min)
python rf_probe.py                         # look-ahead check
python stream64.py                         # streaming == offline
python causal_check.py                     # reconstruction + locality

python gen_prep.py                         # tokens + CLIP embeddings
python gen_ar.py --seconds 21600 --tag gen_v1   # generator; best checkpoint is kept (early stopping)
python gen_sample.py --tag gen_v1          # 10 s and 60 s samples

python switch_test.py --keep1step          # switching with the rule
python stop_test.py                        # stop on command
```

In the paper, the generator run was stopped at about step 9400; the best validation loss (2.19) was at
step 9000. Generation results use one random seed.

## License

Code: Copyright 2026 Kaiji Hiruma, licensed under the [Apache License 2.0](LICENSE).
The license covers the code and the paper in this repository only; it does not cover the motion data
(HumanML3D / AMASS), which has its own license.

## Limitations

HumanML3D clips are at most 10 s long, so some switches (e.g., to sitting) are slow.
The tokenizer was trained on about 10% of the training frames.
Text alignment has not been evaluated with a standard evaluator, and no comparison with other methods
is made.
