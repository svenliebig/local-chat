from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
import time
import warnings

# ANSI color codes
GRAY = "\033[90m"
RESET = "\033[0m"
RED = "\033[91m"


def summarize_conversation(conversation_history):
    # Initialize summarization model
    summarizer_name = "facebook/bart-large-cnn"
    summarizer_tokenizer = BartTokenizer.from_pretrained(summarizer_name)
    summarizer_model = BartForConditionalGeneration.from_pretrained(summarizer_name)

    # Join conversation history into a single string
    text_to_summarize = " ".join(conversation_history[:-2])  # Exclude the last exchange

    # Tokenize and generate summary
    inputs = summarizer_tokenizer(
        [text_to_summarize], max_length=1024, truncation=True, return_tensors="pt"
    )
    summary_ids = summarizer_model.generate(
        **inputs,
        max_length=64,
        min_length=20,
        num_beams=4,
    )

    summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return f"Summary: {summary}"


def main():
    # Initialize the model and tokenizer
    model_name = "facebook/blenderbot-400M-distill"
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

    print("Welcome! Start chatting with the model. Type 'exit' to quit.\n")

    # Keep track of conversation history
    conversation_history = []

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

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            encoded_input = tokenizer.encode(current_input)

        # If conversation is too long, summarize older parts
        if len(encoded_input) > 128:
            print(f"{RED}> {GRAY}Summarizing conversation history...{RESET}")
            # Keep the last exchange (last user input and bot response) as is
            recent_messages = (
                conversation_history[-2:]
                if len(conversation_history) > 2
                else conversation_history
            )
            # Summarize the rest
            summary = summarize_conversation(conversation_history)
            conversation_history = [summary] + recent_messages
            current_input = " ".join(conversation_history)
            encoded_input = tokenizer.encode(current_input)
            print(
                f"{RED}> {GRAY}Conversation summarized (new token length: {len(encoded_input)}){RESET}"
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
