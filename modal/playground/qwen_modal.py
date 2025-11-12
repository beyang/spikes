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
    "fastapi",
)


# Create a class to run on the A100 instance
@app.cls(
    image=image,
    gpu="A100",
    timeout=600,
    min_containers=1,
)
class QwenModel:
    """Class to run Qwen 2.5 7B on Modal."""

    @modal.enter()
    def setup(self):
        """Initialize the model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-7B"

        # Load tokenizer and model
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
        )

    @modal.method()
    def run(self, prompt: str) -> str:
        """Run a prompt through Qwen 2.5 7B."""
        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # Generate response
        print(f"Generating response for prompt: {prompt}")
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            top_p=0.9,
            temperature=0.6,
        )

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        return response


@app.local_entrypoint()
def main():
    """Entry point to run the script locally."""
    prompt = "What is the capital of France? After emitting your answer, write the string '### THE END ###."
    print(f"Prompt: {prompt}")
    print("-" * 50)

    result = QwenModel().run.remote(prompt)
    print(f"Response:\n{result}")
