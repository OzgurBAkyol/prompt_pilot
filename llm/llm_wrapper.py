# llm/llm_wrapper.py

from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from llm.prompt_template import get_prompt_template
from config import CONFIG

def load_llm_chain():
    prompt = get_prompt_template()
    llm = HuggingFaceHub(repo_id=CONFIG["llm_model_name"], model_kwargs={"temperature": 0.7})
    return LLMChain(prompt=prompt, llm=llm)
