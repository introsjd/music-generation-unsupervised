import torch
import numpy as np


def generate_music_transformer(model, seed_tokens,
                                gen_length=256, temperature=1.0):
    model.eval()
    generated = list(seed_tokens)
    with torch.no_grad():
        for _ in range(gen_length):
            context     = generated[-63:]
            inp         = torch.tensor([context], dtype=torch.long)
            logits      = model(inp)
            next_logits = logits[0, -1] / temperature
            probs       = torch.softmax(next_logits, dim=-1)
            next_token  = torch.multinomial(probs, 1).item()
            generated.append(next_token)
    return generated
