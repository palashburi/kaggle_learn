import hellaswag
hellaswag.DATA_CACHE_DIR = '/Users/palashmalu/Desktop/kaggle_learn'


import torch
import torch.nn.functional as F
import tiktoken
import sys
import os

sys.path.append('/Users/palashmalu/Desktop/kaggle_learn')
from self import GPT, GPTConfig
from hellaswag import render_example, iterate_examples

device = 'mps'

def get_most_likely_row(tokens, mask, logits):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    flat_logits  = shift_logits.view(-1, shift_logits.size(-1))
    flat_tokens  = shift_tokens.view(-1)
    loss_per_token = F.cross_entropy(flat_logits, flat_tokens, reduction='none')
    loss_per_token = loss_per_token.view(tokens.size(0), -1)
    shift_mask = mask[..., 1:].contiguous()
    masked_loss = (loss_per_token * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
    return masked_loss.argmin().item()


def run_eval(model, name, num_samples=1000):
    model.eval()
    model.to(device)

    print(f"loading all hellaswag val examples...")
    all_examples = list(iterate_examples("val"))
    total = len(all_examples)
    print(f"total examples in dataset: {total}")

    # evenly spaced across full dataset
    indices = [int(i * total / num_samples) for i in range(num_samples)]
    selected = [all_examples[i] for i in indices]
    print(f"selected {len(selected)} evenly spaced examples")
    print(f"starting eval for: {name}\n")

    num_correct = 0

    for i, example in enumerate(selected):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask   = mask.to(device)

        with torch.no_grad():
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                logits, _ = model(tokens)

        pred = get_most_likely_row(tokens, mask, logits)
        num_correct += int(pred == label)

        # print progress every 100
        if (i + 1) % 100 == 0:
            acc_so_far = num_correct / (i + 1)
            print(f"  [{name}] {i+1}/{num_samples} | "
                  f"correct: {num_correct} | "
                  f"acc: {acc_so_far:.4f} ({acc_so_far*100:.1f}%)")

    final_acc = num_correct / num_samples
    print(f"\n  [{name}] DONE")
    print(f"  correct: {num_correct}/{num_samples}")
    print(f"  accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)\n")
    return final_acc, num_correct


# ── run both evals ──

print("="*60)
print(" MODEL 1 — YOUR TRAINED SHAKESPEARE MODEL")
print("="*60)
model_yours = GPT(GPTConfig())
model_yours.load_state_dict(
    torch.load('shakespeare_gpt.pt', map_location=device)
)
acc_yours, correct_yours = run_eval(model_yours, "shakespeare_gpt10_")

# free memory
del model_yours
torch.mps.empty_cache()
print("memory freed, loading GPT-2...\n")

print("="*60)
print(" MODEL 2 — PRETRAINED GPT-2 (from OpenAI)")
print("="*60)
model_gpt2 = GPT.from_pretrained('gpt2')
acc_gpt2, correct_gpt2 = run_eval(model_gpt2, "gpt2_pretrained")

# ── comparison table ──
print("="*60)
print(" FINAL COMPARISON  (1000 examples)")
print("="*60)
print(f"  random chance:            250/1000 = 25.00%")
print(f"  your shakespeare model:   {correct_yours}/1000 = {acc_yours*100:.2f}%")
print(f"  pretrained GPT-2:         {correct_gpt2}/1000 = {acc_gpt2*100:.2f}%")
print(f"")
print(f"  GPT-2 beats random by:    {(acc_gpt2-0.25)*100:.2f}%")
print(f"  your model beats random:  {(acc_yours-0.25)*100:.2f}%")
print(f"  GPT-2 beats your model:   {(acc_gpt2-acc_yours)*100:.2f}%")
print("="*60)