# test_baseline_run.py

from utils.test_baseline import get_llm_answer

if __name__ == "__main__":
    q = "ACME şirketinin CEO'su kimdir?"
    print("🧠 RL öncesi cevap:", get_llm_answer(q))
