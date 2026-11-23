from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.langchain import CallbackHandler
from langchain.messages import HumanMessage,AIMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
 
load_dotenv()
langfuse_handler = CallbackHandler()

class CustomResponse(BaseModel):
    msg: str = Field(description="The short answer")
    reference: str | None = None

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.3,
    timeout=15,
    max_tokens=1500,
    max_retries=3
)
#conversation = [
    #SystemMessage("You are software security expert who provide precise answers"),
    #HumanMessage("What is binance?"),
    #AIMessage("Binance is a popular bitcoin exchange company"),
    #HumanMessage("Was it hacked recently?")
#]
llm_with_structure = llm.with_structured_output(CustomResponse)
context_binance  = ""
 
with open("binance_hack.md", "r") as fi:
    all_lines = fi.readlines()
    for this_line in all_lines:
        context_binance += this_line + "\n"
 
conversaton_template = ChatPromptTemplate([
    ("system", "You are an assistant to answer with the provided context: {context}"),
    ("human","{question}")
])    
 
#conversation = conversaton_template.format_messages(
   # context=context_binance,
   # question="Was binance hackeed recently?"

#)

 
def validate(resp: CustomResponse) -> CustomResponse:
    if not resp.reference:
        return CustomResponse(msg =  "sorry. no reference", reference=None)
    return resp
 
chain =  conversaton_template | llm_with_structure | RunnableLambda(validate)
 
dict_input = {  "context":context_binance,    "question":"How much was lost in Nvidia because of Jenson Huang?" }
response = chain.invoke(dict_input,
                      config={"callbacks": [langfuse_handler],
                              "metadata" :
                              {
                                  "langfuse_session_id": "binance_session_1",
                                  "langfuse_user_id" :"Nalla"
                              }
                              })
 


 
print(response)