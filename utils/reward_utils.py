# utils/reward_utils.py

from sentence_transformers import SentenceTransformer, util

_embedder_cache = {}

def get_embedder(name):
    if name not in _embedder_cache:
        _embedder_cache[name] = SentenceTransformer(name)
    return _embedder_cache[name]

def compute_similarity(generated, expected, model_name):
    model = get_embedder(model_name)
    emb1 = model.encode(generated, convert_to_tensor=True)
    emb2 = model.encode(expected, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2)[0][0].item()
    return score
