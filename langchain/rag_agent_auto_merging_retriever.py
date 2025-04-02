import re
import sys

from ollama import embed
from regex import R
try:
    from llama_index.core.readers import SimpleDirectoryReader
except ImportError as e:
    print("[ERROR] Required package `llama-index-readers-file` is not installed.")
    print("Please install it using `pip install llama-index-readers-file`.")
    sys.exit(1)

from llama_index.core.llama_dataset import LabelledRagDataset
# from llama_index.core import VectorStoreIndex
from langchain_ollama import ChatOllama, OllamaLLM  # Use ChatOllama for llama3.1 or llama3.2 models
from llama_index.core.evaluation import DatasetGenerator, QueryResponseDataset

from llama_index.core import Document
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext

from llama_index.core.indices.list import ListIndex  # Import ListIndex
from llama_index.core.evaluation import RetrieverEvaluator
import llama_index.core.base as base
from llama_index.core.base import base_retriever

from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.query_engine import JSONalyzeQueryEngine
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core.query_engine import PandasQueryEngine  
from llama_index.core.query_engine import RetrieverQueryEngine

# from llama_index.core.retrievers import SimpleRetriever
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.retrievers import TransformRetriever
from llama_index.core.evaluation.retrieval.metrics import (
    Precision,
    Recall,
)  # Import valid metrics
import nest_asyncio
import llama_index.llms as llms
import llama_index.embeddings as embeddings # import BaseEmbedding # TODO: fix!

nest_asyncio.apply()

# Explicitly state that no OpenAI models, API keys, or embedding/tokenizers are used
print("[INFO] No OpenAI models, API keys, or embedding/tokenizers are used in this script.")

# Load the RAG dataset
file_path = "/Users/yinray/Documents/workspace-mac/llama_index/data/Uber10KDataset2021_rag_dataset.json"
rag_dataset = LabelledRagDataset.from_json(file_path)

# Load documents from the source directory
doc_path = "/Users/yinray/Documents/workspace-mac/llama_index/data/source_files"
try:
    documents = SimpleDirectoryReader(input_dir=doc_path).load_data()  # Ensure `.load_data()` is available
    print(f"[INFO] Loaded {len(documents)} documents from the dataset.")
except AttributeError as e:
    print("[ERROR] Method `.load_data()` not found for `SimpleDirectoryReader`. Please check the library version.")
    sys.exit(1)

# Parse nodes from documents
node_parser = HierarchicalNodeParser.from_defaults()
nodes = node_parser.get_nodes_from_documents(documents)

print("\n[INFO] HierarchicalNodeParser: 'nodes' length = ", len(nodes))

# Get leaf and root nodes
leaf_nodes = get_leaf_nodes(nodes)
print("\n[INFO] No. of 'leaf' nodes that don't have children of their own = ", len(leaf_nodes))
root_nodes = get_root_nodes(nodes)

# Define storage context
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)
storage_context = StorageContext.from_defaults(docstore=docstore)

# Build an index using ChatOllama for embeddings
llm = ChatOllama(model="llama3.1", temperature=0.8, keep_alive=True)  # Use llama3.1 or llama3.2
from langchain_ollama import OllamaLLM  
model = OllamaLLM(model="llama3.2", temperature=0.8, keep_alive=True)  # Use OllamaLLM for llama3.2
model.invoke("Come up with 10 names for a song about parrots")

# Build an index using ListIndex as an alternative to VectorStoreIndex
base_index = ListIndex.from_documents(
    documents=documents
)  # Use ListIndex for simpler indexing
print("[INFO] ListIndex built successfully.")

# Use AutoMergingRetriever
base_retriever = base_index.as_retriever(similarity_top_k=6)
retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)

# Query the retriever
query_str = (
    "What could be the potential outcomes of adjusting the amount of safety"
    " data used in the RLHF stage?"
)

nodes = retriever.retrieve(query_str)
base_nodes = base_retriever.retrieve(query_str)

print("\n[INFO] length of retriever nodes =", len(nodes))  # Example output: 3
print("\n[INFO] length of base_retriever nodes =", len(base_nodes))  # Example output: 6

from llama_index.core.response.notebook_utils import display_source_node

for node in nodes:
    print("\n[INFO] Retrieved node content:")
    print(node.node.get_content())  # Display the content of each retrieved node
    print(display_source_node(node, source_length=10000))  # Display the source node content, if in a notebook environment

# Create a query engine from the index
query_engine = base_index.as_query_engine(llm=llm)  # Use the same LLM for query engine
query_engine_0 = base_index.as_query_engine(llm=model) 
base_query_engine = query_engine  # Ensure this is a valid QueryEngine object

# Generate a query graph using the query engine
query_graph = {
    "root_query_engine": query_engine,  # Map the root query engine
    "child_query_engine": base_query_engine,  # Example child query engine
}

# Obtain the root ID of the query graph
root_id = "root_query_engine"  # Explicitly set the root ID to match the key in query_graph

# Validate the query engine type
if not isinstance(query_engine, BaseQueryEngine):
    raise TypeError("query_engine must be an instance of BaseQueryEngine.")

# Dynamically generate additional query engines if needed
# [NOTE] from transform_query_engine.py:
#     query_engine (BaseQueryEngine): A query engine object.
#     query_transform (BaseQueryTransform): A query transform object.
# transform_query_engine = TransformQueryEngine(base_query_engine)
transform_query_engine = TransformQueryEngine(query_engine, query_transform=None)
query_graph["transform_query_engine"] = transform_query_engine

# Update root_id if necessary
root_id = "root_query_engine"  # Keep the root ID consistent with the query graph

# Maintain a mapping of query_engine_id to the query_engine object
query_engine_mapping = {
    "unique_query_engine_id": query_engine
}
retrieved_query_engine = query_engine_mapping["unique_query_engine_id"]

# Dynamically generate a query graph and extract its root ID
query_graph = {
    "root_query_engine": query_engine,  # Map the root query engine
    "child_query_engine": base_query_engine,  # Example child query engine (can be extended)
}

retriever_dict_example = {}
retriever_dict_example_keys = ["retriever_id"]
retriever_dict_example.setdefault("retriever_id", "unique_retriever_id")
# add("retriever_id", query_engine_mapping["unique_query_engine_id"])
# retriever_dict_example.add("retriever_id", query_engine_mapping["unique_query_engine_id"])
# Ensure unique IDs for retriever and query engine
retriever_id = "retriever_id"
retriever_dict_example.update({"retriever_id": "unique_retriever_id"})
query_engine_id = "unique_query_engine_id" # TODO: fix this w/ retrieved_query_engine.?
# Initialize RecursiveRetriever with a valid retriever_dict
retriever = BaseRetriever(retriever_id, retriever_dict=retriever_dict_example)

query_engine = query_engine

# Define a valid retriever_dict with a base retriever
retriever_dict = {
    "base_retriever": QueryFusionRetriever(
        llm=llm,
        retrievers={},  # Provide an empty retriever or valid retriever objects
        root_id="base_root",
        retriever_dict={},  # Provide a valid retriever_dict
    )
}

# Define a valid query_engine_dict
query_engine_dict = query_graph  # Use the dynamically generated query graph

# Instantiate the RecursiveRetriever
recursive_retriever = RecursiveRetriever(
    root_id=root_id,  # Use the dynamically generated root ID
    retriever_dict=retriever_dict,  # Provide the valid retriever_dict
    query_engine_dict=query_engine_dict,  # Provide the valid query_engine_dict
)

# Instantiate the QueryFusionRetriever with required arguments
dummy_retriever = QueryFusionRetriever(
    llm=llm,
    retrievers=recursive_retriever,
    root_id="fusion_root",  # Provide a valid root_id
    retriever_dict={"recursive": recursive_retriever},  # Provide a valid retriever_dict
)

# Evaluate using the RetrieverEvaluator
rag_evaluator_pack = RetrieverEvaluator(
    rag_dataset=rag_dataset,
    query_engine=query_engine,
    retriever=dummy_retriever,  # Provide a retriever
    metrics=["precision", "recall"],  # Specify evaluation metrics
    show_progress=True,
)

# Example: Run the evaluation (adjust batch_size and sleep_time_in_seconds as needed)
benchmark_df = rag_evaluator_pack.run(
    batch_size=10,  # Number of queries to process in a batch
    sleep_time_in_seconds=1,  # Sleep time between batches
)
print("[INFO] RAG evaluation completed.")
print(benchmark_df)


# # Create a query engine from the index
# # query_engine = index.as_query_engine()
# # query_engine_0 = index_0.as_query_engine()

# # Evaluate using the RetrieverEvaluator
# rag_evaluator_pack = RetrieverEvaluator(
#     rag_dataset=rag_dataset,
#     query_engine=query_engine,
#     show_progress=True,
# )

# # Example: Run the evaluation (adjust batch_size and sleep_time_in_seconds as needed)
# benchmark_df = rag_evaluator_pack.run(
#     batch_size=10,  # Number of queries to process in a batch
#     sleep_time_in_seconds=1,  # Sleep time between batches
# )
# print("[INFO] RAG evaluation completed.")
# print(benchmark_df)