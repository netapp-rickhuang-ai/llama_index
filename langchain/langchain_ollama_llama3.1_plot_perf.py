# # alternatively
from langchain_ollama.llms import OllamaLLM
from langchain_core.runnables import RunnableLambda, Runnable
from datetime import datetime, timezone
import time
import asyncio
import matplotlib.pyplot as plt  # Add this import for plotting

model = OllamaLLM(model="llama3.1")

model.invoke(
    "Come up with 10 names for a song about parrots"
)

def format_t(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )

def convert_to_milliseconds(duration_ns: int) -> float:
    """Convert nanoseconds to milliseconds."""
    return duration_ns / 1_000_000

# Example usage
total_duration_ns = 3700125252
load_duration_ns = 22190161
prompt_eval_duration_ns = 376000000

print(f"[EXAMPLE]Total Duration: {convert_to_milliseconds(total_duration_ns)} ms ")
print(f"[EXAMPLE]Load Duration: {convert_to_milliseconds(load_duration_ns)} ms")
print(
    f"[EXAMPLE]Prompt Eval Duration: {convert_to_milliseconds(prompt_eval_duration_ns)} ms"
)

async def test_runnable(time_to_sleep: int):
    print(f"Runnable[{time_to_sleep}s]: starts at {format_t(time.time())}")
    await asyncio.sleep(time_to_sleep)
    print(f"Runnable[{time_to_sleep}s]: ends at {format_t(time.time())}")

async def fn_start(run_obj: Runnable):
    print(f"on start callback starts at {format_t(time.time())}")
    await asyncio.sleep(3)
    print(f"on start callback ends at {format_t(time.time())}")

# Use the model (example usage)
response = model.generate(
    ["Who is Manny Pacquiao? write in 50 words"]
)  # Pass a list instead of a single string
print(response)

# Extract durations from `generation_info` in the response
generation_info = response.generations[0][0].generation_info
response_total_duration_ns = generation_info.get("total_duration", 0)
response_load_duration_ns = generation_info.get("load_duration", 0)
response_prompt_eval_duration_ns = generation_info.get("prompt_eval_duration", 0)

# Calculate durations in milliseconds
total_duration_ms = convert_to_milliseconds(response_total_duration_ns)
load_duration_ms = convert_to_milliseconds(response_load_duration_ns)
prompt_eval_duration_ms = convert_to_milliseconds(response_prompt_eval_duration_ns)

# Print durations
print(f"[RESPONSE]Total Duration: {total_duration_ms} ms")
print(f"[RESPONSE]Load Duration: {load_duration_ms} ms")
print(f"[RESPONSE]Prompt Eval Duration: {prompt_eval_duration_ms} ms")

# Plot durations
durations = [total_duration_ms, load_duration_ms, prompt_eval_duration_ms]
labels = ["Total Duration", "Load Duration", "Prompt Eval Duration"]

plt.figure(figsize=(8, 6))
plt.bar(labels, durations, color=["blue", "orange", "green"])
plt.xlabel("Duration Type")
plt.ylabel("Duration (ms)")
plt.title("Response Durations in Milliseconds")
plt.legend([f"Model: {model.model}"], loc="upper right")  # Add legend with model name
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# # Define models to compare
# models = ["llama3.2:1b", "deepseek-r1:latest", "phi4-mini"]
# from typing import Dict, List

# durations_data: Dict[str, List[float]] = {model_name: [] for model_name in models}

# # Generate responses and collect durations for each model
# for model_name in models:
#     try:
#         model = OllamaLLM(model=model_name)
#         response = model.generate(
#             ["Who is Manny Pacquiao? write in 50 words"]
#         )  # Example prompt
#         generation_info = response.generations[0][0].generation_info

#         # Extract durations in milliseconds
#         durations_data[model_name] = [
#             convert_to_milliseconds(generation_info.get("total_duration", 0)),
#             convert_to_milliseconds(generation_info.get("load_duration", 0)),
#             convert_to_milliseconds(generation_info.get("prompt_eval_duration", 0)),
#         ]
#     except Exception as e:
#         print(f"Error with model '{model_name}': {e}")
#         durations_data[model_name] = [0, 0, 0]  # Default to 0 for unavailable models

# # Prepare data for plotting
# labels = ["Total Duration", "Load Duration", "Prompt Eval Duration"]
# x = range(len(labels))  # X-axis positions for the labels
# width = 0.15  # Width of each bar

# plt.figure(figsize=(12, 8))

# # Plot durations for each model
# for idx, model_name in enumerate(models):
#     plt.bar(
#         [pos + idx * width for pos in x],  # Offset bars for each model
#         durations_data[model_name],
#         width=width,
#         label=model_name,
#     )

# # Add labels and legend
# plt.xticks([pos + (len(models) - 1) * width / 2 for pos in x], labels)  # Center x-ticks
# plt.xlabel("Duration Type")
# plt.ylabel("Duration (ms)")
# plt.title("Comparison of Response Durations Across Models")
# plt.legend(title="Models", loc="upper right")
# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.show()

# async def fn_end(run_obj: Runnable):
#     print(f"on end callback starts at {format_t(time.time())}")
#     await asyncio.sleep(2)
#     print(f"on end callback ends at {format_t(time.time())}")

# runnable = RunnableLambda(test_runnable).with_alisteners(
#     on_start=fn_start, on_end=fn_end
# )

# async def concurrent_runs():
#     await asyncio.gather(runnable.ainvoke(2), runnable.ainvoke(3))

# asyncio.run(concurrent_runs())
