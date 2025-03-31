from langchain.prompts.chat import ChatPromptTemplate  # Updated import for compatibility
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit  # Corrected import
from langchain_community.tools.json.tool import JsonSpec  # Corrected import
from langchain_ollama import ChatOllama  # Updated import to avoid deprecation
from langchain_community.agent_toolkits.json.base import create_json_agent  # Import create_json_agent
import json  # Use Python's built-in JSON module

# Load the JSON file
file = "/Users/yinray/Documents/workspace-mac/llama_index/data/Uber10KDataset2021_rag_dataset.json"  # Adjust the path to your JSON file
with open(file, "r", encoding="utf-8") as f:
    data = json.load(f)  # Use json.load to parse the JSON file

# Debugging information
print(f"\n[DEBUG] Loaded JSON data from file: {file}")
print(f"\n[INFO] JSON data loaded successfully. Number of examples: {len(data.get('examples', []))}")

# Define the JSON schema and toolkit
json_spec = JsonSpec(dict_=data, max_value_length=4000)  # Create a JSON schema
json_toolkit = JsonToolkit(spec=json_spec)  # Create a LangChain-processable JSON object

# Debugging information for JSON schema
print(f"\n[INFO] JSON schema keys: {list(data.keys())}")
if "examples" in data:
    print(f"[INFO] Number of queries in examples: {len(data['examples'])}")
    for i, example in enumerate(data['examples'][:5]):  # Print first 5 examples for debugging
        # print(f"\n[DEBUG] Example {i+1}: Query: {example.get('query', 'N/A')}, Answer: {example.get('Reference_Answer', 'N/A')}")
        # Answer: 'N/A' for all examples, indicating example.get('Reference_Answer') is not found, do the following instead:
        print(f"\n[DEBUG] Example {i+1}: Query: {example.get('query', 'N/A')}, Answer: {example.get('reference_answer', 'N/A')}")

# Define the prompt
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
         You are a knowledgeable JSON wizard who knows everything about deployments. The below are the JSON descriptions:
         Json Key descriptions in knowledge base:
             - query: The question or query being asked.
             - reference_contexts: Contextual information related to the query.
             - reference_answer: The answer to the query based on the context.
         """),
        ("human", "{input}"),
    ]
)

# Create the JSON agent
json_agent_executor = create_json_agent(
    llm=ChatOllama(model="llama3.1", temperature=0.8),  # Updated to use ChatOllama
    toolkit=json_toolkit,
    prompt=final_prompt,
)

# Execute the agent
output = json_agent_executor.invoke({"input": "What is the state of incorporation for Uber Technologies, Inc.?"})
print(output)