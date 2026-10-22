from langchain_ollama import ChatOllama
from src.config import Configuration

config = Configuration()

def OllamaModel(modelName:str):
    llm = ChatOllama(
    model=modelName,
    temperature=0.3,
    base_url=config.OLLAMA_BASE_URL
    )

    return llm