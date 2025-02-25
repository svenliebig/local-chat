from transformers import pipeline


def main():
    # Initialize the pipeline
    conversation_pipeline = pipeline(
        "text-generation", model="facebook/blenderbot-400M-distill"
    )

    print("Welcome! Start chatting with the model. Type 'exit' to quit.\n")

    # Keep track of conversation history
    conversation_history = []

    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Add user input to history and get response
        conversation_history.append({"role": "user", "content": user_input})
        response = conversation_pipeline(user_input, max_length=100)

        # Extract the bot's response and add it to history
        bot_message = response[0]["generated_text"]
        conversation_history.append({"role": "assistant", "content": bot_message})

        print(f"Bot: {bot_message}\n")


if __name__ == "__main__":
    main()
