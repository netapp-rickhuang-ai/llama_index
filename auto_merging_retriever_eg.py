from pathlib import Path

from llama_index.readers.file import PyMuPDFReader
from llama_index.core import Document
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import AutoMergingRetriever
from langchain_ollama import ChatOllama  # Use ChatOllama for llama3.1 or llama3.2 models
from llama_index.core.indices.list import ListIndex  # Import ListIndex

# Load PDF document
loader = PyMuPDFReader()
docs0 = loader.load(file_path=Path("./data/llama2.pdf"))

# Combine document content
doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]

# Parse nodes from documents
node_parser = HierarchicalNodeParser.from_defaults()
nodes = node_parser.get_nodes_from_documents(docs)

print("\n[INFO] HierarchicalNodeParser: 'nodes' length = ", len(nodes))

# Get leaf and root nodes
leaf_nodes = get_leaf_nodes(nodes)
print("\n[INFO] No. of 'leaf' nodes that don't have children of their own = ", len(leaf_nodes))
root_nodes = get_root_nodes(nodes)

# Define storage context
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)
storage_context = StorageContext.from_defaults(docstore=docstore)

# Use ChatOllama as the LLM
llm = ChatOllama(model="llama3.1", temperature=0.8, keep_alive=True)

# # Load index into vector index
# base_index = VectorStoreIndex(
#     leaf_nodes,
#     storage_context=storage_context,
# )

base_index = ListIndex.from_documents(
    documents=docs
)  # Use ListIndex for simpler indexing
print("[INFO] ListIndex `index` built successfully.")


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