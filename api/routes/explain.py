from LLM.prompt import Prompt
from LLM.llmModel import OllamaModel
from src.config import Configuration

config = Configuration()

llmModelName = config.OLLAMA_MODEL_NAME

def llmExplaination(predicted_json:dict,predicted_Type_fail:dict) -> str:
    
    Prompting = Prompt(predicted_json,predicted_Type_fail)
    
    llm = OllamaModel(llmModelName)
    
    chain = Prompting | llm
    
    question = "Based on what info you have Explain in 2 Sentence Exactly what cause the Failure of Machine"
    
    response = chain.invoke({
        "Question":question,
        "Predction_json":predicted_json,
        "Predicton_Type_json":predicted_Type_fail
        })
    
    return response.content
