import textwrap
from dotenv import load_dotenv

# Gemini AI client imports
from google import genai
from google.genai.errors import APIError

# Load environment variables from .env file
load_dotenv()

# Main model for chatbot responses
Main_model = "gemini-2.5-flash"

# Summary model for summarization tasks
Summary_model = "gemini-2.0-flash-lite"

# Maximum detailed turns to keep
MAX_RECENT_TURNS = 3

# System instructions
SYSTEM_INSTRUCTION = (
    "You are a friendly, helpful terminal chatbot specializing in computer science and technical topics. "
    "Respond concisely in 1–2 lines. "
    "Use your own knowledge to answer questions, and use the context summary only to maintain conversation continuity."
)

# Summary prompt template
SUMMARY_PROMPT_TEMPLATE = textwrap.dedent("""
    Generate a concise, 1-2 lines summary of the conversation below.
    Focus only on main topics, important details, and any expressed user preferences.
    Do not mention the prompt or instructions in your output.

    Conversation History (User:Model turns):
    ---
    {history}
    ---
    Summary:
""")

# Format conversation history for prompts(user:model)
def format_history(history):
  return "\n".join([f"User: {turn['user']}\nBot: {turn['bot']}" for turn in history])

# Summary of the conversation history
def generate_summary(client, history):
    if not history:
        return "No prior conversation.", history

    formatted_history = format_history(history)
    summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(history=formatted_history)

    try:
        response = client.models.generate_content(
            model=Summary_model,
            contents=[summary_prompt]
        )

        summary = response.text.strip()
        trimmed_history = history[-MAX_RECENT_TURNS:]

        return summary, trimmed_history

    except APIError as e:
        print(f"API Error during summarization: {e}")
        return "Summary unavailable.", history

# Build the context for the main model
def build_context(summary, history):
  recent_text = format_history(history)
  return textwrap.dedent(f"""
        {SYSTEM_INSTRUCTION}

        Conversation Summary:
        {summary}

        Recent Turns:
        {recent_text}
    """)

# Main chatbot function
def chatbot():
  try:
    # Client autometically gets the API key from the GEMINI_API_KEY from .env
    client = genai.Client()
  except Exception as e:
    raise ValueError("Could not initialize Gemini client. Check GEMINI_API_KEY") from e

  print("-" * 100)
  print(" Gemini computer science chatbot ")
  print("-" * 100)
  print(f"\nMain model: {Main_model}")
  print(f"Summary model: {Summary_model}\n")
  print("Type '\quit' or '\exit' to end the chat.")

  conversation_history = []
  summary, history = generate_summary(client, conversation_history)

  while True:
    try:
      user_input = input("\nYou: ").strip()

      if user_input.lower() in ['\quit', '\exit']:
        print("Exiting chat. Goodbye!")
        break

      if not user_input:
        print("Please enter a valid message.")
        continue

      context = build_context(summary, history)

      prompt = f"{context}\nUser: {user_input}"

      response = client.models.generate_content(
        model= Main_model,
        contents = [prompt]
      )

      model_response = response.text
      print(f"Bot: {model_response}")

      history.append({"user": user_input, "bot": model_response})

      summary, history = generate_summary(client, history)

      # Uncomment below line to see conversation summary after each turn
      # print(f"\n[Conversation Summary]:{summary}")

    except Exception as e:
      print(f"An error occurred: {e}")
      break

if __name__ == "__main__":
  chatbot()
