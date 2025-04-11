# env/qa_env.py

import gym
from gym import spaces
import json

from utils.reward_utils import compute_similarity
from llm.llm_wrapper import load_llm_chain

class QAEnv(gym.Env):
    def __init__(self, context_path, qa_path, embed_model):
        super(QAEnv, self).__init__()
        self.context = open(context_path).read()
        self.qa_pairs = json.load(open(qa_path))
        self.embed_model = embed_model
        self.chain = load_llm_chain()
        self.index = 0

    def reset(self):
        self.index = 0
        return self.qa_pairs[self.index]["question"]

    def step(self, _):
        question = self.qa_pairs[self.index]["question"]
        expected = self.qa_pairs[self.index]["answer"]

        result = self.chain.run(context=self.context, question=question)
        reward = compute_similarity(result, expected, self.embed_model)

        self.index += 1
        done = self.index >= len(self.qa_pairs)

        return question, reward, done, {"generated": result}
