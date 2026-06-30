"""
validate_embeddings_py.py — Python side of the Rhobots embedding validation.

Called by validate_embeddings.R.  Loads texts from a CSV, embeds them using
HuggingFace transformers (AutoModel + mean pool + L2 norm), and writes the
embedding matrix to a CSV file.

The pipeline deliberately mirrors what Rhobots does in R:
  1. AutoModel.from_pretrained  (same weights as load_hf_bert)
  2. AutoTokenizer               (same tokenizer.json)
  3. Mean pool over non-padding tokens
  4. L2 normalise rows
  5. Add instruction prefix when supplied (same as embed_texts(prefix=...))

Usage (called automatically by validate_embeddings.R):
  python validate_embeddings_py.py \\
      --model   sentence-transformers/all-MiniLM-L6-v2 \\
      --input   /path/to/texts.csv   \\
      --output  /path/to/out.csv     \\
      --prefix  ""                   \\
      --col     text                 \\
      --max_len 256                  \\
      --batch   32
"""

import argparse, sys, os
import numpy as np
import pandas as pd

def mean_pool(hidden_states, attention_mask):
    """Masked mean pool over the sequence dimension."""
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

def l2_norm(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms

def embed(model, tokenizer, texts, prefix, max_len, batch_size, device):
    import torch
    model.eval()
    model.to(device)
    if prefix:
        texts = [prefix + t for t in texts]
    all_vecs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        hidden = out.last_hidden_state          # (B, L, H)
        mask   = enc["attention_mask"]          # (B, L)
        pooled = mean_pool(hidden, mask)        # (B, H)
        all_vecs.append(pooled.cpu().numpy())
        print(f"  embedded {min(start + batch_size, len(texts))} / {len(texts)}",
              flush=True)
    return l2_norm(np.vstack(all_vecs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True)
    p.add_argument("--input",   required=True)
    p.add_argument("--output",  required=True)
    p.add_argument("--prefix",  default="")
    p.add_argument("--col",     default="text")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch",   type=int, default=32)
    args = p.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        sys.exit(f"Missing Python dependency: {e}\n"
                 "Install with: pip install transformers torch")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Python] Model:  {args.model}")
    print(f"[Python] Device: {device}")
    print(f"[Python] Prefix: '{args.prefix}'")

    df = pd.read_csv(args.input)
    if args.col not in df.columns:
        sys.exit(f"Column '{args.col}' not found. Available: {list(df.columns)}")
    texts = df[args.col].astype(str).tolist()
    print(f"[Python] Texts:  {len(texts)}")

    print("[Python] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModel.from_pretrained(args.model)

    print("[Python] Embedding...")
    emb = embed(model, tokenizer, texts, args.prefix,
                args.max_len, args.batch, device)

    print(f"[Python] Matrix: {emb.shape[0]} x {emb.shape[1]}")
    pd.DataFrame(emb).to_csv(args.output, index=False)
    print(f"[Python] Saved to: {args.output}")


if __name__ == "__main__":
    main()
