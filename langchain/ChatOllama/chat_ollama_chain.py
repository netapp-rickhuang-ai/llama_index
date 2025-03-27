# # alternatively
# from langchain_ollama.llms import OllamaLLM
from langchain_core.runnables import RunnableLambda, Runnable
from datetime import datetime, timezone
# import time
import asyncio
import matplotlib.pyplot as plt  # Add this import for plotting
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1",
    temperature=0.8
    # other params...
)

from langchain_core.messages import AIMessage

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print("\n[DEBUG] ai_msg.content: ", ai_msg.content)

# Access and print usage metadata
try:
    usage_metadata = ai_msg.usage_metadata  # Directly access the `usage_metadata` attribute
    if usage_metadata:
        total_tokens = usage_metadata.get("total_tokens", "N/A")
        prompt_tokens = usage_metadata.get("prompt_tokens", "N/A")
        completion_tokens = usage_metadata.get("completion_tokens", "N/A")
        print(f"[DEBUG] usage_metadata: Total Tokens: {total_tokens}, Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}")
    else:
        print("[DEBUG] usage_metadata: No metadata available")
except AttributeError as e:
    print(f"[ERROR] Unable to access usage_metadata: {e}")


