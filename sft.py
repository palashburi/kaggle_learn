import torch
import torch.nn.functional as F

def bce_from_scratch(logits,targets,eps = 1e-7):

    probs = torch.sigmoid(logits)
    probs = torch.clamp(probs,eps,1-eps)
    
    loss  = -( targets*torch.log(probs) + (1-targets)*torch.log(1-probs))
    return loss.mean()

logits = torch.randn(10,requires_grad=True)
targets = torch.randint(0,2,(10,)).float()

my_loss = bce_from_scratch(logits,targets)
torch_loss = F.binary_cross_entropy_with_logits(logits,targets)

print(f"mine: {my_loss.item(): .6f}")
print(f"Torch: {torch_loss.item(): .6f}")
assert torch.isclose(my_loss,torch_loss,atol = 1e-5)

