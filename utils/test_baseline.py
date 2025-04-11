# utils/test_baseline.py

from llm.llm_wrapper import load_llm_chain
from config import CONFIG

def get_llm_answer(question: str) -> str:
    """
    LangChain üzerinden RL uygulanmamış haliyle cevap döner.
    """
    with open(CONFIG["context_path"], "r", encoding="utf-8") as f:
        context = f.read()

    chain = load_llm_chain()
    return chain.run(context=context, question=question)
