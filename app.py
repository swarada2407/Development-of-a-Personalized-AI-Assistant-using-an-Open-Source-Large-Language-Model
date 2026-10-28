from __future__ import annotations

from pathlib import Path
from threading import Lock

import torch
from flask import Flask, jsonify, render_template, request
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

app = Flask(__name__)

ADAPTER_DIR = Path(__file__).resolve().parent / "personalized_lora_adapter"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer clearly and directly. When a user provides source material, "
    "base your answer on that material and do not invent details."
)
MODEL_LOCK = Lock()
model: PreTrainedModel | None = None
tokenizer: PreTrainedTokenizerBase | None = None


def load_model() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    global model, tokenizer

    if model is None or tokenizer is None:
        with MODEL_LOCK:
            if model is None or tokenizer is None:
                adapter_config = PeftConfig.from_pretrained(ADAPTER_DIR, local_files_only=True)
                loaded_tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, local_files_only=True)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                loaded_base = AutoModelForCausalLM.from_pretrained(
                    adapter_config.base_model_name_or_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                )
                loaded_model = PeftModel.from_pretrained(
                    loaded_base,
                    ADAPTER_DIR,
                    is_trainable=False,
                )
                loaded_model.to(device)
                loaded_model.eval()
                tokenizer = loaded_tokenizer
                model = loaded_model

    return model, tokenizer


def generate_reply(prompt: str) -> str:
    current_model, current_tokenizer = load_model()
    tokenized = current_tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = tokenized["input_ids"].to(current_model.device)

    with torch.inference_mode():
        output_ids = current_model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=current_tokenizer.eos_token_id,
        )

    answer_ids = output_ids[0, input_ids.shape[-1] :]
    return current_tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("message", "")).strip()
    if not prompt:
        return jsonify({"error": "Enter a message to start chatting."}), 400

    try:
        reply = generate_reply(prompt)
    except OSError as error:
        app.logger.exception("Could not load the base model or LoRA adapter.")
        return jsonify({"error": f"Could not load the AI model: {error}"}), 503

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
