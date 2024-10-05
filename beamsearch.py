import torch as t
import torch.nn as nn


def beam_search(model: nn.Module, max_depth: int, n_beams: int) -> t.Tensor:
    start_token_ids = [0] * n_beams 
    stopped_beams_count = 0
    results = [{"cumprob": 1, "token_ids": []} for _ in range(n_beams)]
    local_results = []
    while max_depth > 0 and stopped_beams_count < n_beams:
        for start_token_id in start_token_ids:
            # generate one more sequence
            # TODO: What about sequences > 1 token?
            # model.forward("The") -> ["cat", "dog"]
            # model.forward("The").forward("cat")
            # model.forward("cat") =/= model.forward("The cat")
            # model.forward(["the", "cat"]) ?
            probs = model.forward(start_token_id) # token/vocab size
            top_k_probs, top_k_indices = probs.topk(n_beams, dim=-1)
            local_max = [{"prob": top_k_probs[i].item(),
                          "token_id": top_k_indices[i].item()}
                          for i in range(n_beams)]
            local_results.append(local_max)
            # store the top n_beams in a list of lists
        # Compare and store step
        # 1. For each local results, combine with results 
        # multiply by cumprob
        conditional_probabilities = [] # top_k ** 2
        # TODO: Do not remove ended sequence
        # [{"cumprob": 0.3, "token_ids": [1], "ended": False}] 
        top_k_conditional = sorted(conditional_probabilities,
                                   key=lambda x: x.get("cumprob"))[::-1][:n_beams]
        # [{"cumprob": 0.3, "token_ids": [1]}]
        # Decide on the next start tokens
        start_token_ids = [x["token_ids"][-1]
                           for x in top_k_conditional
                           if x != END_TOKEN]
        # Kill sequences that have hit the stop token
        # update results, stopped_beams_count
            # consider waiting till end for stopped_beams_count as an option
        max_depth -= 1


