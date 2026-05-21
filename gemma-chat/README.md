# gemma-chat — Gemma 4 E4B on Perlmutter (pm-gpu)

A minimal chat UI (Gradio) backed by `google/gemma-4-E4B-it` running on a single
A100 GPU node at NERSC. Accessed from your laptop over an SSH port-forward.

## Layout

```
gemma-chat/
├── app.py            # Gradio chat app, streams from transformers
├── requirements.txt  # torch (cu124) + transformers + gradio
├── setup_env.sh      # one-shot uv venv install (run on a login node)
├── interactive.sh    # salloc a GPU and run app.py in foreground
├── serve.sbatch      # long-running sbatch job
└── README.md
```

## Prereqs

- NERSC account with a GPU allocation (project ending in `_g`).
- A Hugging Face token with access to `google/gemma-4-E4B-it`
  (Gemma models are gated — accept the license on the model page once,
  then `export HF_TOKEN=hf_xxx`).
- `sshproxy` set up locally so you can `ssh perlmutter.nersc.gov` without
  re-typing MFA each time. See
  [`docs.nersc.gov/connect/mfa/#sshproxy`](https://docs.nersc.gov/connect/mfa/#sshproxy).

## One-time setup (login node)

```bash
ssh perlmutter.nersc.gov
git clone <this repo> && cd <repo>/gemma-chat
bash setup_env.sh
```

This installs `uv` (if missing) and creates a venv at
`$SCRATCH/envs/gemma-chat` with torch + transformers + gradio. Set `VENV_DIR`
to put it elsewhere.

## Run interactively (recommended for prototyping)

From a login node:

```bash
export HF_TOKEN=hf_xxx
export ACCOUNT=m9999_g       # your GPU project
bash interactive.sh
```

`salloc` grants one A100 within ~6 minutes
([interactive QOS docs](https://docs.nersc.gov/jobs/interactive/#interactive-qos-on-perlmutter)),
the script activates the venv and launches `app.py` on port 7860.

It prints something like:

```
>>> from your laptop:  ssh -L 7860:nid001234:7860 perlmutter.nersc.gov
>>> then open:         http://localhost:7860
```

Open a second terminal on your laptop, paste that `ssh -L …` command, leave
it running, and browse <http://localhost:7860>.

## Run as a longer batch job

Edit `serve.sbatch` to set `#SBATCH -A <account>_g`, then:

```bash
export HF_TOKEN=hf_xxx
sbatch serve.sbatch
tail -f gemma-chat-<jobid>.log    # find the node name + tunnel command
```

The job uses the `shared` QOS with 1 GPU, per
[Perlmutter running-jobs guidance](https://docs.nersc.gov/systems/perlmutter/running-jobs/#1-node-1-task-1-gpu)
("Jobs using 1 or 2 GPUs should request the shared QOS").

## Knobs

Override via env vars:

| var                     | default                  | meaning                          |
|-------------------------|--------------------------|----------------------------------|
| `GEMMA_MODEL`           | `google/gemma-4-E4B-it`  | swap to E2B, 26B, 31B, etc.      |
| `GEMMA_PORT`            | `7860`                   | Gradio port (forward this one)   |
| `GEMMA_HOST`            | `0.0.0.0`                | bind addr inside the compute node |
| `GEMMA_MAX_NEW_TOKENS`  | `1024`                   | per-turn generation cap          |
| `HF_HOME`               | `$SCRATCH/hf-cache`      | model weight cache               |
| `VENV_DIR`              | `$SCRATCH/envs/gemma-chat` | install location               |

## Notes & caveats

- E4B-it fits in ~8 GB bf16, so a single A100-40G is overkill. Bumping to the
  27B/31B model just means switching `GEMMA_MODEL` and using `-C "gpu&hbm80g"`
  in `serve.sbatch` (and possibly more GPUs).
- `app.py` uses `AutoProcessor` so it also accepts images/audio if you extend
  the Gradio UI — the underlying Gemma 4 model is multimodal. The default
  `gr.ChatInterface` here is text-only.
- This setup is single-user. To share with a colleague, either (a) have them
  open the same `ssh -L` tunnel, or (b) flip `share=False` to `share=True`
  in `app.py` to get a temporary `*.gradio.live` public URL — fine for a
  demo, not for anything sensitive.
- Pulling weights the first time is ~8 GB; do it once on a login node by
  running `huggingface-cli download $GEMMA_MODEL` so the compute-node job
  doesn't waste GPU-hours on the download.
