# llm/prompt_template.py

from langchain.prompts import PromptTemplate

def get_prompt_template():
    template = """Sen {context} konusunda uzmansın. Şu soruya cevap ver:
Soru: {question}
Cevap:"""
    return PromptTemplate(template=template, input_variables=["context", "question"])
