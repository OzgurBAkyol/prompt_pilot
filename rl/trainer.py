# rl/trainer.py

from env.qa_env import QAEnv
from config import CONFIG

def train_agent(episodes=10):
    env = QAEnv(CONFIG["context_path"], CONFIG["qa_path"], CONFIG["embedding_model"])

    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0

        done = False
        while not done:
            # Sabit aksiyon (LLM output'u)
            obs, reward, done, info = env.step(None)
            total_reward += reward

        print(f"[Episode {ep+1}] Total Reward: {total_reward:.4f}")
        print("Son cevap:", info["generated"])
