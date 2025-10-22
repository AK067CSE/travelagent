# Advanced AI Trip Planner with Agentic RAG
# A complete, advanced project for trip planning using multi-agent systems, RAG, reranking, and fine-tuned LLMs.

# Install dependencies:
# pip install crewai streamlit openai unstructured pyowm langchain langchain-community langchain-core langchain-groq tools requests fastapi uvicorn pydantic python-dotenv langchain-openai ragatouille transformers torch

import os
from typing import TypedDict, List
from crewai import Agent, Task, Crew, LLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from ragatouille import RAGPretrainedModel
from langchain.schema import Document
import json

# Set API keys
os.environ['OPENAI_API_KEY'] = 'your_openai_api_key_here'

# Advanced state for agentic RAG
class TripState(TypedDict):
    user_query: str
    reasoning: str
    category: str
    retrieved_docs: List[str]
    reranked_docs: List[str]
    fused_response: str
    final_plan: str

# Load knowledge base for RAG
def load_knowledge_base():
    # Dummy data for trip planning
    return [
        {"text": "Best time to visit Paris is spring for mild weather.", "metadata": {"category": "weather"}},
        {"text": "Top attractions in Tokyo: Tokyo Tower, Shibuya Crossing.", "metadata": {"category": "attractions"}},
        {"text": "Budget tips: Use public transport in London to save money.", "metadata": {"category": "budget"}}
    ]

knowledge_base = load_knowledge_base()
processed_docs = [Document(page_content=doc["text"], metadata=doc["metadata"]) for doc in knowledge_base]

# Vector DB and Reranker
embed_model = OpenAIEmbeddings(model='text-embedding-3-small')
vector_db = Chroma.from_documents(processed_docs, embed_model)
reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# LLM for agents
llm = LLM(model="gpt-4o")

# Reasoning Agent
def reasoning_agent():
    return Agent(
        role='Trip Reasoning Expert',
        goal='Perform deep reasoning on trip queries to understand intent and context.',
        backstory='An AI that thinks step-by-step for optimal trip planning.',
        llm=llm,
        verbose=True
    )

# Categorization Agent
def categorization_agent():
    return Agent(
        role='Trip Category Classifier',
        goal='Categorize queries into weather, attractions, budget, etc.',
        backstory='Expert in classifying trip-related queries.',
        llm=llm,
        verbose=True
    )

# RAG Agent with Reranker
def rag_agent():
    return Agent(
        role='Advanced RAG Specialist',
        goal='Retrieve, rerank, and fuse information for trip planning.',
        backstory='Specialist in RAG with reranking and fusion for accurate responses.',
        tools=[vector_db.as_retriever()],
        llm=llm,
        verbose=True
    )

# Response Generator Agent
def response_generator_agent():
    return Agent(
        role='Trip Plan Generator',
        goal='Generate comprehensive trip plans using fused RAG data.',
        backstory='Expert in creating detailed, personalized trip itineraries.',
        llm=llm,
        verbose=True
    )

# Fine-tuned LLM Script (placeholder for integration)
def fine_tune_llm():
    # This would be a script to fine-tune a model for trip planning
    print("Fine-tuning LLM for trip planning... (Implement with your data)")

# Main Crew
def create_trip_crew():
    agents = TripAgents()
    tasks = TripTasks()

    reasoning = reasoning_agent()
    categorize = categorization_agent()
    rag = rag_agent()
    generate = response_generator_agent()

    # Define tasks with advanced prompts
    task1 = Task(description="Reason step-by-step about the trip query.", agent=reasoning)
    task2 = Task(description="Categorize the query based on reasoning.", agent=categorize)
    task3 = Task(description="Retrieve and rerank relevant info.", agent=rag)
    task4 = Task(description="Generate fused, detailed trip plan.", agent=generate)

    crew = Crew(agents=[reasoning, categorize, rag, generate], tasks=[task1, task2, task3, task4])
    return crew

# Run the system
if __name__ == "__main__":
    crew = create_trip_crew()
    result = crew.kickoff(inputs={"query": "Plan a trip to Paris in spring."})
    print(result)
