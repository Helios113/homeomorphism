import hashlib
import torch
from homeomorphism import models
from homeomorphism.interventions import load_model_for_baseline, build_prepared_input

def model_hash(m):
    h = hashlib.sha256()
    for k, v in sorted(m.model.state_dict().items()):
        h.update(k.encode('utf-8'))
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
base = models.load_model('gpt2', weights='trained', seed=0, device=device)
h_base = model_hash(base)
text = 'The quick brown fox jumps over the lazy dog.'

for b in ['topological_initialisation', 'maximum_entropy_injection', 'syntactic_disintegration', 'semantic_scrambling']:
    mm = load_model_for_baseline(model_name='gpt2', weights='trained', baseline=b, seed=0, device=device)
    h_new = model_hash(mm)
    prep = build_prepared_input(m=mm, text=text, max_tokens=8, baseline=b, root_seed=0, sample_index=0)
    print('---', b)
    print('weights changed:', h_new != h_base)
    print('has input_ids:', 'input_ids' in prep.forward_kwargs)
    print('has inputs_embeds:', 'inputs_embeds' in prep.forward_kwargs)
    print('mask shape:', tuple(prep.forward_kwargs['attention_mask'].shape))
