# # alternatively
from langchain_ollama.llms import OllamaLLM
from langchain_core.runnables import RunnableLambda, Runnable
from datetime import datetime, timezone
import time
import asyncio

model = OllamaLLM(model="llama3.2:1b")

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
    ["Hello, how are you?"]
)  # Pass a list instead of a single string
print(response)

# Extract durations from `generation_info` in the response
generation_info = response.generations[0][0].generation_info
response_total_duration_ns = generation_info.get("total_duration", 0)
response_load_duration_ns = generation_info.get("load_duration", 0)
response_prompt_eval_duration_ns = generation_info.get("prompt_eval_duration", 0)

# Calculate and display durations in milliseconds
print(f"[RESPONSE]Total Duration: {convert_to_milliseconds(response_total_duration_ns)} ms")
print(f"[RESPONSE]Load Duration: {convert_to_milliseconds(response_load_duration_ns)} ms")
print(f"[RESPONSE]Prompt Eval Duration: {convert_to_milliseconds(response_prompt_eval_duration_ns)} ms")

async def fn_end(run_obj: Runnable):
    print(f"on end callback starts at {format_t(time.time())}")
    await asyncio.sleep(2)
    print(f"on end callback ends at {format_t(time.time())}")


runnable = RunnableLambda(test_runnable).with_alisteners(
    on_start=fn_start, on_end=fn_end
)

async def concurrent_runs():
    await asyncio.gather(runnable.ainvoke(2), runnable.ainvoke(3))


asyncio.run(concurrent_runs())
