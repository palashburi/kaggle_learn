import re
import torch
import json
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

from datetime import datetime

def parse_transcript(file_path, min_words=5):
    text_segments = []
    images = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    image_pattern = r'\[(image_\d+)\s+(.+)\]'
    current_timestamp = None
    current_text = ""
    last_timestamp = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if ':' in line and '[' in line:
            ts_match = re.match(r'(\d+):(\d+):', line)
            img_match = re.search(image_pattern, line)
            if ts_match and img_match:
                minutes = int(ts_match.group(1))
                seconds = int(ts_match.group(2))
                timestamp = minutes * 60 + seconds
                image_id = img_match.group(1)
                description = img_match.group(2).strip()
                images.append({
                    "timestamp": float(timestamp),
                    "path": f"{image_id}.jpg",
                    "description": description
                })
            continue

        ts_match = re.match(r'(\d+):(\d+)', line)
        if ts_match:
            minutes = int(ts_match.group(1))
            seconds = int(ts_match.group(2))
            current_timestamp = minutes * 60 + seconds
            
            if current_text.strip() and len(current_text.split()) >= min_words:
                text_segments.append({
                    "timestamp": float(last_timestamp if last_timestamp else current_timestamp),
                    "text": current_text.strip()
                })
            
            current_text = ""
            last_timestamp = current_timestamp
            continue

        current_text += " " + line

    if current_text.strip() and len(current_text.split()) >= min_words and last_timestamp is not None:
        text_segments.append({
            "timestamp": float(last_timestamp),
            "text": current_text.strip()
        })

    return text_segments, images

def filter_meaningful_segments(segments, min_words=8, exclude_patterns=None):
    if exclude_patterns is None:
        exclude_patterns = [r'^SPEAKER\s+\d+$', r'^\[.*\]$']
    
    filtered = []
    for seg in segments:
        text = seg['text'].strip()
        word_count = len(text.split())
        
        if word_count < min_words:
            continue
            
        exclude = False
        for pattern in exclude_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                exclude = True
                break
                
        if not exclude:
            filtered.append(seg)
            
    return filtered

def save_results_to_json(results, filename="image_text_matches.json"):
    """Save the matching results to a JSON file"""
    output_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_images": len(results),
            "sigma_values_used": list(results.keys())
        },
        "results": results
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {filename}")
    return filename

if __name__ == "__main__":
    transcript_file = "transcript.txt"
    text_segments, images = parse_transcript(transcript_file, min_words=5)
    
    meaningful_segments = filter_meaningful_segments(text_segments, min_words=8)
    
    print(f"Original segments: {len(text_segments)}")
    print(f"Meaningful segments: {len(meaningful_segments)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)

    texts = [seg["text"] for seg in meaningful_segments]
    if not texts:
        print("No meaningful text segments found. Exiting.")
        raise SystemExit(0)

    text_inputs = processor(text=texts, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_embeddings = model.get_text_features(**text_inputs)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    image_embeddings = []
    image_timestamps = []
    valid_images = []
    for img_entry in images:
        try:
            img = Image.open(img_entry["path"]).convert("RGB")
        except FileNotFoundError:
            continue

        image_inputs = processor(images=img, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        with torch.no_grad():
            img_embed = model.get_image_features(**image_inputs)
        img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)

        image_embeddings.append(img_embed)
        image_timestamps.append(float(img_entry["timestamp"]))
        valid_images.append(img_entry)

    if not image_embeddings:
        print("No valid images loaded. Exiting.")
        raise SystemExit(0)

    image_embeddings = torch.cat(image_embeddings, dim=0)
    image_timestamps = torch.tensor(image_timestamps, device=device)
    text_timestamps = torch.tensor([seg["timestamp"] for seg in meaningful_segments], device=device)

    similarity = image_embeddings @ text_embeddings.T

    sigma_values = [15.0, 30.0, 45.0, 60.0]
    all_results = {}
    
    for sigma in sigma_values:
        print(f"\n{'='*60}")
        print(f"RESULTS WITH SIGMA = {sigma}s")
        print(f"{'='*60}")
        
        time_diff = torch.cdist(image_timestamps.unsqueeze(1), text_timestamps.unsqueeze(1), p=2)
        decay = torch.exp(- (time_diff ** 2) / (2 * (sigma ** 2)))
        final_similarity = similarity * decay

        topk = 3
        topk_vals, topk_idx = torch.topk(final_similarity, k=min(topk, final_similarity.size(1)), dim=1)
        
        sigma_results = {}
        
        for i, img_meta in enumerate(valid_images):
            image_key = img_meta['path']
            sigma_results[image_key] = {
                "image_timestamp": float(img_meta['timestamp']),
                "image_description": img_meta['description'],
                "top_matches": []
            }
            
            for rank in range(topk_vals.size(1)):
                score = topk_vals[i, rank].item()
                idx = topk_idx[i, rank].item()
                text_segment = meaningful_segments[idx]
                time_gap = abs(img_meta['timestamp'] - text_segment['timestamp'])
                
                match_data = {
                    "rank": rank + 1,
                    "score": float(score),
                    "text_timestamp": float(text_segment['timestamp']),
                    "time_gap_seconds": float(time_gap),
                    "text_content": text_segment['text'],
                    "word_count": len(text_segment['text'].split())
                }
                
                sigma_results[image_key]["top_matches"].append(match_data)
            
            print(f"\nImage: {image_key} (t={img_meta['timestamp']:.1f}s)")
            for match in sigma_results[image_key]["top_matches"]:
                print(f"  #{match['rank']}: score={match['score']:.4f} @ {match['text_timestamp']:.1f}s (gap: {match['time_gap_seconds']:.1f}s)")
        
        all_results[f"sigma_{int(sigma)}"] = sigma_results

    # Save all results to JSON
    json_filename = save_results_to_json(all_results)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY: Top matches stored in JSON format")
    print(f"{'='*60}")
    for sigma_key, sigma_data in all_results.items():
        print(f"\nFor {sigma_key}:")
        for image_name, image_data in sigma_data.items():
            best_match = image_data["top_matches"][0]
            print(f"  {image_name}: Best match @ {best_match['text_timestamp']:.1f}s (score: {best_match['score']:.4f})")

    print(f"\nDetailed results saved to: {json_filename}")
    print("Done.")


# import re
# import torch
# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image

# def parse_transcript(file_path, min_words=5):
#     text_segments = []
#     images = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()

#     image_pattern = r'\[(image_\d+)\s+(.+)\]'
#     current_timestamp = None
#     current_text = ""
#     last_timestamp = None

#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue

#         if ':' in line and '[' in line:
#             ts_match = re.match(r'(\d+):(\d+):', line)
#             img_match = re.search(image_pattern, line)
#             if ts_match and img_match:
#                 minutes = int(ts_match.group(1))
#                 seconds = int(ts_match.group(2))
#                 timestamp = minutes * 60 + seconds
#                 image_id = img_match.group(1)
#                 description = img_match.group(2).strip()
#                 images.append({
#                     "timestamp": float(timestamp),
#                     "path": f"{image_id}.jpg",
#                     "description": description
#                 })
#             continue

#         ts_match = re.match(r'(\d+):(\d+)', line)
#         if ts_match:
#             minutes = int(ts_match.group(1))
#             seconds = int(ts_match.group(2))
#             current_timestamp = minutes * 60 + seconds
            
#             if current_text.strip() and len(current_text.split()) >= min_words:
#                 text_segments.append({
#                     "timestamp": float(last_timestamp if last_timestamp else current_timestamp),
#                     "text": current_text.strip()
#                 })
            
#             current_text = ""
#             last_timestamp = current_timestamp
#             continue

#         current_text += " " + line

#     if current_text.strip() and len(current_text.split()) >= min_words and last_timestamp is not None:
#         text_segments.append({
#             "timestamp": float(last_timestamp),
#             "text": current_text.strip()
#         })

#     return text_segments, images

# def filter_meaningful_segments(segments, min_words=8, exclude_patterns=None):
#     if exclude_patterns is None:
#         exclude_patterns = [r'^SPEAKER\s+\d+$', r'^\[.*\]$']
    
#     filtered = []
#     for seg in segments:
#         text = seg['text'].strip()
#         word_count = len(text.split())
        
#         if word_count < min_words:
#             continue
            
#         exclude = False
#         for pattern in exclude_patterns:
#             if re.match(pattern, text, re.IGNORECASE):
#                 exclude = True
#                 break
                
#         if not exclude:
#             filtered.append(seg)
            
#     return filtered

# if __name__ == "__main__":
#     transcript_file = "transcript.txt"
#     text_segments, images = parse_transcript(transcript_file, min_words=5)
    
#     meaningful_segments = filter_meaningful_segments(text_segments, min_words=8)
    
#     print(f"Original segments: {len(text_segments)}")
#     print(f"Meaningful segments: {len(meaningful_segments)}")
#     print("=== Meaningful Text Segments ===")
#     for seg in meaningful_segments[:10]:
#         print(f"[{seg['timestamp']:.2f}s] {seg['text'][:100]}...")
#     print()

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     model = model.to(device)

#     texts = [seg["text"] for seg in meaningful_segments]
#     if not texts:
#         print("No meaningful text segments found. Exiting.")
#         raise SystemExit(0)

#     text_inputs = processor(text=texts, return_tensors="pt", padding=True)
#     text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
#     with torch.no_grad():
#         text_embeddings = model.get_text_features(**text_inputs)
#     text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

#     image_embeddings = []
#     image_timestamps = []
#     valid_images = []
#     for img_entry in images:
#         try:
#             img = Image.open(img_entry["path"]).convert("RGB")
#         except FileNotFoundError:
#             continue

#         image_inputs = processor(images=img, return_tensors="pt")
#         image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
#         with torch.no_grad():
#             img_embed = model.get_image_features(**image_inputs)
#         img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)

#         image_embeddings.append(img_embed)
#         image_timestamps.append(float(img_entry["timestamp"]))
#         valid_images.append(img_entry)

#     if not image_embeddings:
#         print("No valid images loaded. Exiting.")
#         raise SystemExit(0)

#     image_embeddings = torch.cat(image_embeddings, dim=0)
#     image_timestamps = torch.tensor(image_timestamps, device=device)
#     text_timestamps = torch.tensor([seg["timestamp"] for seg in meaningful_segments], device=device)

#     similarity = image_embeddings @ text_embeddings.T

#     sigma_values = [15.0, 30.0, 45.0, 60.0]
    
#     for sigma in sigma_values:
#         print(f"\n{'='*60}")
#         print(f"RESULTS WITH SIGMA = {sigma}s")
#         print(f"{'='*60}")
        
#         time_diff = torch.cdist(image_timestamps.unsqueeze(1), text_timestamps.unsqueeze(1), p=2)
#         decay = torch.exp(- (time_diff ** 2) / (2 * (sigma ** 2)))
#         final_similarity = similarity * decay

#         best_matches = torch.argmax(final_similarity, dim=1)

#         print("\n=== Best Matches ===")
#         for i, img_meta in enumerate(valid_images):
#             best_idx = best_matches[i].item()
#             score = final_similarity[i, best_idx].item()
#             txt_meta = meaningful_segments[best_idx]
#             time_gap = abs(img_meta['timestamp'] - txt_meta['timestamp'])
            
#             print(f"Image: {img_meta['path']} (t={img_meta['timestamp']:.1f}s)")
#             print(f"  → Text @ {txt_meta['timestamp']:.1f}s (gap: {time_gap:.1f}s, score: {score:.4f})")
#             print(f"  Content: \"{txt_meta['text'][:80]}...\"")
#             print()

#         topk = 3
#         topk_vals, topk_idx = torch.topk(final_similarity, k=min(topk, final_similarity.size(1)), dim=1)
        
#         print("=== Top Matches Summary ===")
#         for i, img_meta in enumerate(valid_images):
#             print(f"\nImage {i+1}: {img_meta['path']} (t={img_meta['timestamp']:.1f}s)")
#             for rank in range(topk_vals.size(1)):
#                 s = topk_vals[i, rank].item()
#                 idx = topk_idx[i, rank].item()
#                 ts = meaningful_segments[idx]['timestamp']
#                 time_gap = abs(img_meta['timestamp'] - ts)
#                 txt = meaningful_segments[idx]['text']
#                 print(f"  #{rank+1}: score={s:.4f} @ {ts:.1f}s (gap: {time_gap:.1f}s) → {txt[:60]}...")

#     print("\nDone.")

