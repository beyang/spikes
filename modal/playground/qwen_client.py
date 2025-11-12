"""
Client script to interact with the deployed Qwen model.
"""

import modal


def main():
    """Interactive client for the Qwen model."""
    QwenModel = modal.Cls.from_name("qwen-2.5-7b", "QwenModel")
    model = QwenModel()

    print("Connected to Qwen 2.5 7B. Type 'exit' or press Ctrl+C to quit.\n")

    while True:
        try:
            prompt = input("\n>>> ").strip()

            if not prompt or prompt.lower() == "exit":
                print("Exiting...")
                break

            print("\nGenerating response...")
            result = model.run.remote(prompt)
            print(f"\n{result}")
            print("-" * 80)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
