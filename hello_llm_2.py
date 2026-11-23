from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.langchain import CallbackHandler
from langchain.messages import HumanMessage,AIMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from ReadactManager import RedactManager
from ChunkManager import ChunkManager
from RetrieveManager import RetrieveManager
from StoreManager import StoreManager
from config import load_config
from prompts import get_prmpt

load_dotenv()
 
prompt_det = get_prmpt("binance_qa",2)
config = load_config()
chunkMan = ChunkManager(config)
storeMan = StoreManager()
retrieveMan = RetrieveManager(config)
langfuse_handler = CallbackHandler()
 
 
def debug_step(data):
    print("...inspecting...")
    return data
 
 
context_binance = ""
with open("binance_hack.md", "r") as fi:
    all_lines = fi.readlines()
    for this_line in all_lines:
        context_binance += this_line + "\n"
RedactMan = RedactManager()
redacted_context = RedactMan.performRedacting_Of_PII(context_binance)
chunks = chunkMan.chunk_markdown(redacted_context)
documents, vector_store = storeMan.embed_and_store(chunks)
question = "How much was lost in the binance hack?"
search_results = retrieveMan.query_hybrid(question, documents, vector_store)
final_result = retrieveMan.rerank_documents(question, search_results)

class CustomReponse(BaseModel):
    msg: str = Field(description="The short answer")
    reference: str | None = None
    confidence: int = Field(
        ge=0,
        le=100)

prompt = ChatPromptTemplate.from_template(prompt_det["system"])

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)

llm_with_struture = llm.with_structured_output(CustomReponse)

#chain = prompt | llm_with_struture

parser = StrOutputParser()
upper_case_convertor = RunnableLambda(lambda txt: txt.upper())
chain  = prompt | llm_with_struture
 
#chain  = prompt | llm_with_struture    #|  parser |  upper_case_convertor 
chain  = prompt | llm | parser
 
def ask_question(quest: str):    
    search_results = retrieveMan.query_hybrid(quest, documents, vector_store)
    final_result = retrieveMan.rerank_documents(quest, search_results)
    response = chain.invoke({
                "question": quest,
                "ctxt":final_result
                    },        config={"callbacks": [langfuse_handler],
                                "metadata" :
                                {
                                    "langfuse_session_id": "binance_session_1",
                                    "langfuse_user_id" :"Nalla",
                                    "prompt_name": prompt_det["name"],
                                    "prompt_version": prompt_det["version"],
                                }})
    # return response
    trace_id = langfuse_handler.last_trace_id
    print(f"............... {trace_id}")
    return {
        "answer": response,
        "retrieval_context": final_result[0].page_content,
        "trace_id": trace_id
    }

#print(response)