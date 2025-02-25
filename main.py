from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import time

# ANSI color codes
GRAY = "\033[90m"
RESET = "\033[0m"
RED = "\033[91m"

def main():
    # Initialize the model and tokenizer
    model_name = "facebook/blenderbot-400M-distill"
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

    print("Welcome! Start chatting with the model. Type 'exit' to quit.\n")

    # Keep track of conversation history
    conversation_history = []  # Change to list to store multiple exchanges

    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Add current input to conversation history
        conversation_history.append(f"You: {user_input}")
        current_input = " ".join(conversation_history)

        # Time the tokenization process
        token_start = time.time()
        # Check if input is too long and truncate from the beginning if necessary
        encoded_input = tokenizer.encode(current_input)
        if len(encoded_input) > 128:
            # Remove oldest messages until we're under the limit
            while len(encoded_input) > 120 and conversation_history:
                conversation_history.pop(0)  # Remove oldest message
                current_input = " ".join(conversation_history)
                encoded_input = tokenizer.encode(current_input)
            print(
                f"  {RED}>{GRAY}Truncated conversation to fit context window (removed oldest messages, remaining: {len(conversation_history)}){RESET}"
            )

        # Tokenize for generation
        inputs = tokenizer([current_input], return_tensors="pt", truncation=True)
        token_end = time.time()
        print(
            f"{GRAY}Tokenization took {(token_end - token_start)*1000:.2f}ms (Token length: {len(encoded_input)}){RESET}"
        )

        # Time the generation process
        gen_start = time.time()
        reply_ids = model.generate(
            **inputs,
            max_length=128,
            min_length=8,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            num_beams=5,
            no_repeat_ngram_size=2,
        )
        bot_message = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        gen_end = time.time()
        print(f"{GRAY}Generation took {(gen_end - gen_start)*1000:.2f}ms{RESET}")

        # Update conversation history with bot's response
        conversation_history.append(f"Bot: {bot_message}")

        print(f"Bot: {bot_message}\n")


if __name__ == "__main__":
    main()
