# gemma — run Gemma 4 on Perlmutter pm-gpu

A single script that loads `google/gemma-4-E4B-it` on an A100 and either runs
one prompt or drops into a REPL.

## One-time setup (login node)

```bash
ssh perlmutter.nersc.gov
module load python
python -m venv $SCRATCH/envs/gemma
source $SCRATCH/envs/gemma/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 transformers>=4.57 accelerate huggingface_hub sentencepiece
huggingface-cli login    # paste a token with access to gated Gemma weights
```

## Run

```bash
salloc -N 1 -q interactive -t 01:00:00 -C gpu --gpus 1 -A m9999_g
source $SCRATCH/envs/gemma/bin/activate
python run.py "Explain bf16 like I'm five."
# or, for a REPL:
python run.py
```

That's it. ~8 GB on the GPU, first run downloads weights to `~/.cache/huggingface`
(set `HF_HOME=$SCRATCH/hf-cache` to keep it on scratch).
