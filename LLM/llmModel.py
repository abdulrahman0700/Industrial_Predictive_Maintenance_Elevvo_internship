from langchain_ollama import ChatOllama
from src.config import Configuration

config = Configuration()

def OllamaModel(modelName:str):
    """ Ollama defintions and configurations .

    Args:
    modelName (str) : the Local large language model name

    Returns:
    ChatOllama: A configured Ollama chat model instance.

    """
    llm = ChatOllama(
    model=modelName,
    temperature=0.3,
    base_url=config.OLLAMA_BASE_URL
    )

    return llm