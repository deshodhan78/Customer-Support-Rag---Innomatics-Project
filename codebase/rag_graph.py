from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Setup Vector Store & Retriever
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. Setup LLM
from langchain_huggingface import ChatHuggingFace

llm_endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="conversational",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    max_new_tokens=512,
    stop_sequences=["Question:", "Answer:", "</s>", "<|endoftext|>"]
)
llm = ChatHuggingFace(llm=llm_endpoint)

class GraphState(TypedDict):
    question: str
    context: List[str]
    answer: str
    escalate: bool
    intent: str  # "technical" or "account"
    messages: Annotated[list, "add_messages"]

# 3. Node Functions

def classify_intent(state: GraphState):
    print("---CLASSIFYING INTENT---")
    question = state["question"]
    
    prompt = ChatPromptTemplate.from_template("""
    Classify the following user question into one of two categories:
    1. "technical": Questions about router setup, reset, Wi-Fi issues, or technical specs.
    2. "account": Questions about refunds, cancellations, warranty claims, or account status.
    
    Question: {question}
    Category (output only the word "technical" or "account"):""")
    
    chain = prompt | llm
    response = chain.invoke({"question": question})
    intent = response.content.lower().strip() if hasattr(response, "content") else str(response).lower()
    
    # Fallback to technical if unclear
    if "account" not in intent:
        intent = "technical"
    else:
        intent = "account"
        
    return {"intent": intent}

def retrieve(state: GraphState):
    print("---RETRIEVING---")
    question = state["question"]
    docs = retriever.invoke(question)
    context = [doc.page_content for doc in docs]
    return {"context": context}

def grade_documents(state: GraphState):
    print("---GRADING DOCUMENTS---")
    question = state["question"]
    context = state["context"]
    
    if not context:
        return {"intent": "irrelevant"} # Trigger escalation
        
    prompt = ChatPromptTemplate.from_template("""
    You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    
    Retrieved Document: {doc}
    User Question: {question}
    
    Relevant (yes/no):""")
    
    chain = prompt | llm
    
    # We'll grade the first/best doc for simplicity in this flow
    response = chain.invoke({"doc": context[0], "question": question})
    score = response.content.lower().strip() if hasattr(response, "content") else str(response).lower()
    
    if "yes" in score:
        return {"intent": "technical"} # Proceed to generate
    else:
        return {"intent": "account"} # Re-use 'account' intent to trigger escalation branch

def generate(state: GraphState):
    print("---GENERATING---")
    question = state["question"]
    context = "\n".join(state["context"])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a TechNova Support Assistant. Provide a concise answer based ONLY on the context."),
        ("human", "Context: {context}\n\nQuestion: {question}\n\nHelpful Answer:")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    answer = response.content if hasattr(response, "content") else str(response)
    
    # Post-processing
    for stop in ["Question:", "Answer:", "[/ASS]", "</s>"]:
        answer = answer.split(stop)[0].strip()
    
    return {"answer": answer, "escalate": False}

def human_escalate(state: GraphState):
    print("---ESCALATING TO HUMAN---")
    return {
        "answer": "I am sorry, I don't have enough information to help with that. Let me connect you to a human agent.",
        "escalate": True
    }

# 4. Routing Logic
def route_after_classify(state: GraphState):
    if state["intent"] == "account":
        return "human_escalate"
    return "retrieve"

def route_after_grade(state: GraphState):
    if state["intent"] == "account": # Used as a proxy for 'irrelevant'
        return "human_escalate"
    return "generate"

# 5. Build the Graph
workflow = StateGraph(GraphState)

workflow.add_node("classify", classify_intent)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("human_escalate", human_escalate)

workflow.add_edge(START, "classify")

workflow.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "human_escalate": "human_escalate",
        "retrieve": "retrieve"
    }
)

workflow.add_edge("retrieve", "grade")

workflow.add_conditional_edges(
    "grade",
    route_after_grade,
    {
        "human_escalate": "human_escalate",
        "generate": "generate"
    }
)

workflow.add_edge("generate", END)
workflow.add_edge("human_escalate", END)

# Compile
graph = workflow.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "user_123"}}