# # alternatively
# from langchain_ollama.llms import OllamaLLM
from langchain_core.runnables import RunnableLambda, Runnable
from datetime import datetime, timezone
from langchain_core.messages import AIMessage
# import time
import asyncio
import matplotlib.pyplot as plt  # Add this import for plotting
from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="llama3.1",
    temperature=0.8,
    num_predict=256
    # other params...
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print("\n[INPUT] for prompt: ", messages[1][:])
print("\n[OUTPUT] ai_msg.content: ", ai_msg.content)

# Stream for lllama3.1
llm_0 = ChatOllama(
    model="llama3.1",
    temperature=0.5,
    num_predict=64  # Adjust num_predict for streaming
    # other params...
)
messages_0 = [
    (
        "system",
        "You are an Agentic AI that is both helpful and creative. Your task is to return sentences when prompted. ",
    ),
    ("human", "Who is Manny Pacquiao? write in 50 words"),
]
print("\n[INPUT] for prompt: ", messages_0[1][:])

# for chunk in llm_0.stream(messages_0):
#     print("\n[OUTPUT] chunk.text in llm_0: ", chunk.text(), end="")
stream = llm_0.stream(messages_0)
full = next(stream)
for chunk in stream:
    full += chunk
print("\n[OUTPUT] full llm_0.stream text chunk: ", full)

# Save `AIMessageChunk` as `ai_msg_0`
ai_msg_0 = full  # `full` is of type `AIMessageChunk`

# Check the type of `ai_msg_0` for debugging
print(f"[RESULT] Type of ai_msg_0: {type(ai_msg_0)}")  #  Type of ai_msg_0: <class 'langchain_core.messages.ai.AIMessageChunk'>

# Extract durations dynamically from `ai_msg_0`
if hasattr(ai_msg_0, "response_metadata"):
    #generation_info = ai_msg_0.response_metadata.get("total_duration", {})
    total_duration_ns = ai_msg_0.response_metadata.get("total_duration", 0) 
    load_duration_ns = ai_msg_0.response_metadata.get("load_duration", 0)
    eval_duration_ns = ai_msg_0.response_metadata.get("eval_duration", 0)

    # Convert durations to milliseconds
    def convert_to_milliseconds(duration_ns: int) -> float:
        """Convert nanoseconds to milliseconds."""
        return duration_ns / 1_000_000

    total_duration_ms = convert_to_milliseconds(total_duration_ns)
    load_duration_ms = convert_to_milliseconds(load_duration_ns)
    eval_duration_ms = convert_to_milliseconds(eval_duration_ns)

    # Print durations in milliseconds
    print(f"[OUTPUT] Total Duration: {total_duration_ms} ms")
    print(f"[OUTPUT] Load Duration: {load_duration_ms} ms")
    print(f"[OUTPUT] Eval Duration: {eval_duration_ms} ms")
else:
    print("[ERROR] ai_msg_0 does not have 'response_metadata' attribute")

# Plot durations
durations = [total_duration_ms, load_duration_ms, eval_duration_ms]
labels = ["Total Duration", "Load Duration", "Prompt Eval Duration"]

plt.figure(figsize=(8, 6))
plt.bar(labels, durations, color=["blue", "orange", "green"])
plt.xlabel("Duration Type")
plt.ylabel("Duration (ms)")
plt.title("Response Durations in Milliseconds for fixed 'BOXER' prompt")

# Combine both legends into one
plt.legend([f"Model: {llm_0.model}"], loc="upper right", title="API: ChatOllama")

plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()