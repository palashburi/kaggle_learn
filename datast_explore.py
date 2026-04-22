
from datasets import load_dataset

print("loading datasets...")
ds_finemath  = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", streaming=True, split="train")
ds_textbooks = load_dataset("open-phi/textbooks",      streaming=True,   split="train")
ds_stories   = load_dataset("roneneldan/TinyStories",  streaming=True,   split="train")
print("done\n")

def explore(name, ds, n=3):
    print("=" * 60)
    print(name)
    print("=" * 60)
    it = iter(ds)
    for i in range(n):
        ex = next(it)
        text = ex.get('text', ex.get('content', ex.get('markdown', str(ex))))
        print(f"\nsample {i+1}")
        print(f"keys:    {list(ex.keys())}")
        print(f"length:  {len(text)} chars")
        print(f"preview: {text[:300]}")
        print()
    del it

explore("FINEMATH",  ds_finemath)
explore("TEXTBOOKS", ds_textbooks)
explore("TINYSTORIES", ds_stories)

# --------
# from datasets import load_dataset

# print("loading datasets in streaming mode (no full download)...")

# ds_finemath  = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", streaming=True, split="train")
# ds_textbooks = load_dataset("open-phi/textbooks",      streaming=True,   split="train")
# ds_stories   = load_dataset("roneneldan/TinyStories",  streaming=True,   split="train")

# print("done — exploring 3 samples from each\n")


# # ─────────────────────────────────────────
# # EXPLORE FINEMATH
# # ─────────────────────────────────────────

# print("=" * 60)
# print("FINEMATH (finemath-4plus)")
# print("=" * 60)

# finemath_iter = iter(ds_finemath)
# for i in range(3):
#     ex = next(finemath_iter)
#     print(f"\n--- sample {i+1} ---")
#     print(f"keys:    {list(ex.keys())}")
#     # grab whichever text field exists
#     text = ex.get('text', ex.get('content', str(ex)))
#     print(f"length:  {len(text)} chars")
#     print(f"preview:\n{text[:400]}")
    


# # ─────────────────────────────────────────
# # EXPLORE PHI TEXTBOOKS
# # ─────────────────────────────────────────

# print("=" * 60)
# print("PHI TEXTBOOKS (open-phi/textbooks)")
# print("=" * 60)

# textbooks_iter = iter(ds_textbooks)
# for i in range(3):
#     ex = next(textbooks_iter)
#     print(f"\n--- sample {i+1} ---")
#     print(f"keys:    {list(ex.keys())}")
#     text = ex.get('textbook', ex.get('text', str(ex)))
#     print(f"length:  {len(text)} chars")
#     print(f"preview:\n{text[:400]}")
#     print()


# # ─────────────────────────────────────────
# # EXPLORE TINYSTORIES
# # ─────────────────────────────────────────

# print("=" * 60)
# print("TINYSTORIES (roneneldan/TinyStories)")
# print("=" * 60)

# stories_iter = iter(ds_stories)
# for i in range(3):
#     ex = next(stories_iter)
#     print(f"\n--- sample {i+1} ---")
#     print(f"keys:    {list(ex.keys())}")
#     text = ex.get('text', str(ex))
#     print(f"length:  {len(text)} chars")
#     print(f"preview:\n{text[:400]}")
#     print()


# # ─────────────────────────────────────────
# # TOKEN LENGTH DISTRIBUTION
# # ─────────────────────────────────────────

# import tiktoken
# enc = tiktoken.get_encoding('gpt2')

# print("=" * 60)
# print("TOKEN LENGTH CHECK (10 samples each)")
# print("=" * 60)

# datasets_to_check = {
#     "finemath":  (iter(ds_finemath),  lambda ex: ex.get('text', ex.get('content', ''))),
#     "textbooks": (iter(ds_textbooks), lambda ex: ex.get('textbook', ex.get('text', ''))),
#     "stories":   (iter(ds_stories),   lambda ex: ex.get('text', '')),
# }

# for name, (it, get_text) in datasets_to_check.items():
#     lengths = []
#     for _ in range(10):
#         ex = next(it)
#         text = get_text(ex)
#         lengths.append(len(enc.encode(text)))
#     avg = sum(lengths) / len(lengths)
#     print(f"{name:12s} | avg tokens: {avg:.0f} | "
#           f"min: {min(lengths)} | max: {max(lengths)}")