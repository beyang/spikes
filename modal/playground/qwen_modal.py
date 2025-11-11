"""
Modal script to run Qwen 2.5 7B on an A100 instance.
"""

import modal

# Create a Modal app
app = modal.App("qwen-2.5-7b")

# Define the image with necessary dependencies
image = modal.Image.debian_slim().pip_install(
    "transformers>=4.36.0",
    "torch>=2.0.0",
    "accelerate>=0.24.0",
)


# Create a function to run on the A100 instance
@app.function(
    image=image,
    gpu="A100",
    timeout=600,
)
def run_qwen(prompt: str) -> str:
    """Run a prompt through Qwen 2.5 7B."""
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

    model_name = "Qwen/Qwen2.5-7B"

    # Load tokenizer and model
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    # Prepare input
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate response
    print(f"Generating response for prompt: {prompt}")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        top_p=0.9,
        temperature=0.6,
    )

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


@app.local_entrypoint()
def main():
    """Entry point to run the script locally."""
    prompt = "What is the capital of France? After emitting your answer, write the string '### THE END ###."
    print(f"Prompt: {prompt}")
    print("-" * 50)

    result = run_qwen.remote(prompt)
    print(f"Response:\n{result}")
