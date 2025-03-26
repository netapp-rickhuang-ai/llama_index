
# # alternatively
from langchain_ollama.llms import OllamaLLM

model = OllamaLLM(model="llama3.2:1b")
model.invoke(
    "Come up with 10 names for a song about parrots"
)

# Use the model (example usage)
response = model.generate(["Hello, how are you?"])  # Pass a list instead of a single string
print(response)
