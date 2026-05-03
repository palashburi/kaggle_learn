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


def focal_loss(logits,targets,alpha=0.25,gamma=2.0,eps =1e-7,reduction ='mean'):

    bce = F.binary_cross_entropy_with_logits(logits,targets,reduction='none')
    pt = torch.exp(-bce)

    alpha_t = targets*alpha + (1-targets)*(1-alpha)
    focal_weigth = alpha_t *(1-pt)**gamma
    loss = focal_weigth * bce

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
    


logits_easy = torch.tensor([5.0])   # sigmoid(5) ≈ 0.993
targets_easy = torch.tensor([1.0])

logits_hard = torch.tensor([0.1])   # sigmoid(0.1) ≈ 0.525
targets_hard = torch.tensor([1.0])

print(f"Easy BCE:  {F.binary_cross_entropy_with_logits(logits_easy, targets_easy):.4f}")
print(f"Easy Focal: {focal_loss(logits_easy, targets_easy):.6f}")
print(f"Hard BCE:  {F.binary_cross_entropy_with_logits(logits_hard, targets_hard):.4f}")
print(f"Hard Focal: {focal_loss(logits_hard, targets_hard):.4f}")