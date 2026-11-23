
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
 
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from ChunkManager import ChunkManager
from StoreManager import StoreManager
from RetrieveManager import RetrieveManager
 
load_dotenv()
 
REPORT_PATH = "cybersecurity_incident.md"
 
with open(REPORT_PATH, "r", encoding="utf-8") as f:
    markdown_doc = f.read()
 
chunkMan = ChunkManager()
chunks = chunkMan.chunk_markdown(markdown_doc)
storeMan = StoreManager()
 
for i, chunk in enumerate(chunks):
    print(f"...>>>>... chunk num{i} has length {len(chunk)}")
    print(chunk[:100])
 
documents, vector_store = storeMan.embed_and_store(chunks)
# search_results = query_chroma_db("How much money is lost due to the hack?")

search_queries = [
    "What caused the cybersecurity incident?",
    "How many customer records were affected?",
    "What actions were taken to stop the attack?",
    "Was the payment processing service affected?"
]

retrieveMan = RetrieveManager()

for search_query in search_queries:

    print("\n================================")
    print("QUESTION:", search_query)
    print("================================")

    results = retrieveMan.query_hybrid(
        search_query,
        documents=documents,
        vector_store=vector_store,
        n_results=3
    )

    final_results = retrieveMan.rerank_documents(
        search_query,
        results
    )

    for res in final_results:
        print(res.page_content)

 
# for i, res in enumerate(results):
#     print(f"\n*** Result {i} ***")
#     print(f"Content:\n{res.page_content}")