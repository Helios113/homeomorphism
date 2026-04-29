import torch

eps = 0.1
W1 = torch.eye(2)
W2 = eps * torch.eye(2)
b1 = torch.zeros(2)

def block_seq(h_seq):
    # Vectorized for shape (T, d)
    # W1 is (2,2), h_seq is (T,2)
    # For h_seq[0], it was W1 @ h + b1
    # For matrix multiplication, h_seq @ W1.T + b1
    z = h_seq @ W1.T + b1
    return h_seq + torch.relu(z) @ W2.T

h = torch.tensor([1.0, 1.0])
h_seq = h.to(torch.float32).view(1, -1)

from homeomorphism import jacobian
bj, per_diag = jacobian.build_jacobian(
    block_seq, h_seq, scope='diagonal', evaluate='per_diagonal_slogdet'
)
sign, logabs = per_diag[0]
print(logabs.item())
