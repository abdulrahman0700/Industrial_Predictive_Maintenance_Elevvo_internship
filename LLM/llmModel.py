from langchain_ollama import ChatOllama

def OllamaModel(modelName:str):
    llm = ChatOllama(
    model=modelName,
    temperature=0.3  
    )

    return llm