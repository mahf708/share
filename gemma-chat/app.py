"""Gradio chat UI backed by google/gemma-4-E4B-it on a Perlmutter GPU node."""

from __future__ import annotations

import os
import socket
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer

MODEL_ID = os.environ.get("GEMMA_MODEL", "google/gemma-4-E4B-it")
HOST = os.environ.get("GEMMA_HOST", "0.0.0.0")
PORT = int(os.environ.get("GEMMA_PORT", "7860"))
MAX_NEW_TOKENS = int(os.environ.get("GEMMA_MAX_NEW_TOKENS", "1024"))

print(f"[gemma-chat] loading {MODEL_ID} on {socket.gethostname()} ...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print(f"[gemma-chat] ready on device {model.device}", flush=True)


def chat(message: str, history: list[dict], system_prompt: str, temperature: float):
    messages: list[dict] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    streamer = TextIteratorStreamer(
        processor.tokenizer if hasattr(processor, "tokenizer") else processor,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
    )
    Thread(target=model.generate, kwargs=gen_kwargs, daemon=True).start()

    partial = ""
    for token in streamer:
        partial += token
        yield partial


demo = gr.ChatInterface(
    fn=chat,
    type="messages",
    title="Gemma 4 E4B on Perlmutter",
    description=f"Running `{MODEL_ID}` on `{socket.gethostname()}`.",
    additional_inputs=[
        gr.Textbox(
            label="System prompt",
            value="You are a concise, helpful assistant running on NERSC Perlmutter.",
            lines=2,
        ),
        gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature"),
    ],
)

if __name__ == "__main__":
    demo.queue().launch(server_name=HOST, server_port=PORT, share=False)
