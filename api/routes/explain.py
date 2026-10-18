from LLM.prompt import Prompt
from LLM.llmModel import OllamaModel

def llmExplaination(predicted_json,predicted_Type_fail):
    
    Prompting = Prompt(predicted_json,predicted_Type_fail)
    
    llm = OllamaModel("qwen3:0.6b")
    
    chain = Prompting | llm
    
    question = "Based on what info you have Explain in 2 Sentence Exactly what cause the Failure of Machine"
    
    response = chain.invoke({
        "Question":question,
        "Predction_json":predicted_json,
        "Predicton_Type_json":predicted_Type_fail
        })
    
    return response.content
