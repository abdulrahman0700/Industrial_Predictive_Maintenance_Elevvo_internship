from langchain_core.prompts import ChatPromptTemplate

def Prompt(Predction_json,Predicton_Type_json):
    Templete = """
    You are an Reliability Engineer Expert in industrial predictive maintenance. 
    Answer the user's question based on the following data:
    Question: {Question}

    Example JSON structure:
    {{
       "Predction_json": "your_data_here",
       "Predicton_Type_json": "your_type_here"
    }}
    
    CRITICAL: Provide your answer in plain text only. Do NOT use JSON, markdown code blocks, or keys. Just output the exact sentences requested.
    """

    prompt = ChatPromptTemplate.from_template(Templete)

    return prompt
