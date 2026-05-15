"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from typing import Optional

from model import Transformer, make_src_mask, make_tgt_mask


UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]

# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        V = self.vocab_size
        eps = self.smoothing

        # Create smoothed distribution
        # Start: fill every position with eps / (V - 2)
        #        -2 because we exclude the gold token AND the pad token
        smooth_val = eps / (V - 2)
        dist = torch.full_like(logits, smooth_val)     # [B*T, V]

        # Set PAD column to 0 probability
        dist[:, self.pad_idx] = 0.0

        # Set gold token probability to (1 - eps)
        dist.scatter_(1, target.unsqueeze(1), 1.0 - eps)

        # Mask out rows where target is PAD
        # (we don't compute loss on padding positions)
        pad_positions = (target == self.pad_idx)
        dist[pad_positions] = 0.0

        # KL divergence: KL(dist || softmax(logits))
        # = sum(dist * log(dist/softmax)) = sum(dist * (log_dist - log_softmax))
        # Since dist is our target, use nn.KLDivLoss with log_softmax input
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.kl_div(log_probs, dist, reduction='sum')

        # Normalize by number of non-PAD tokens
        n_tokens = (~pad_positions).sum().float()
        return loss / n_tokens.clamp(min=1)


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    model.train() if is_train else model.eval()

    total_loss, total_tokens = 0.0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch_idx, (src, tgt) in enumerate(data_iter):
            src = src.to(device)
            tgt = tgt.to(device)

            # Teacher forcing: input is tgt[:-1], target is tgt[1:]
            tgt_input  = tgt[:, :-1]   # [B, T-1]
            tgt_output = tgt[:, 1:]    # [B, T-1]

            # Build masks
            src_mask = make_src_mask(src)               # [B, 1, 1, src_len]
            tgt_mask = make_tgt_mask(tgt_input)         # [B, 1, T-1, T-1]

            # Forward pass
            logits = model(src, tgt_input, src_mask, tgt_mask)  # [B, T-1, V]

            # Reshape for loss: [B*(T-1), V] and [B*(T-1)]
            B, T, V = logits.shape
            loss = loss_fn(logits.reshape(-1, V), tgt_output.reshape(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping (important for Transformers!)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            # Count non-PAD tokens for normalization
            n_tokens = (tgt_output != 1).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    # TODO: Task 3.3 — implement token-by-token greedy decoding
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)               # [1, src_len, d_model]
        ys = torch.tensor([[start_symbol]], device=device)  # [1, 1]

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys)                   # [1, 1, cur_len, cur_len]
            out = model.decode(memory, src_mask, ys, tgt_mask)  # [1, cur_len, V]
            
            # Take the last position's logits → most likely next token
            next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
            ys = torch.cat([ys, next_token], dim=1)                    # [1, cur_len+1]

            if next_token.item() == end_symbol:
                break

    return ys  


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════
from collections import Counter
import math

def _get_ngrams(tokens, n):
    """Return a Counter of all n-grams in a token list."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def _corpus_bleu(hypotheses, references, max_n=4):
    """
    Compute corpus-level BLEU score (0-100) using only stdlib.

    hypotheses : list of strings (model output sentences)
    references : list of strings (gold sentences, same order)
    max_n      : max n-gram order (default 4, standard BLEU-4)
    """
    # Accumulators for each n-gram order
    clipped_counts = [0] * max_n   # numerator of precision
    total_counts   = [0] * max_n   # denominator of precision

    hyp_len = 0   # total hypothesis token count
    ref_len = 0   # total reference token count (for brevity penalty)

    for hyp_str, ref_str in zip(hypotheses, references):
        hyp_tokens = hyp_str.split()
        ref_tokens = ref_str.split()

        hyp_len += len(hyp_tokens)
        ref_len += len(ref_tokens)

        for n in range(1, max_n + 1):
            hyp_ngrams = _get_ngrams(hyp_tokens, n)
            ref_ngrams = _get_ngrams(ref_tokens, n)

            # Clipped count: for each n-gram in hyp, count up to how many
            # times it appears in the reference (not more)
            for ngram, hyp_count in hyp_ngrams.items():
                clipped_counts[n-1] += min(hyp_count, ref_ngrams.get(ngram, 0))

            total_counts[n-1] += max(len(hyp_tokens) - n + 1, 0)

    # Compute log-average of precisions
    log_avg = 0.0
    for n in range(max_n):
        if clipped_counts[n] == 0 or total_counts[n] == 0:
            return 0.0   # any zero precision → BLEU = 0
        log_avg += (1.0 / max_n) * math.log(clipped_counts[n] / total_counts[n])

    # Brevity penalty
    if hyp_len == 0:
        return 0.0
    bp = min(1.0, math.exp(1 - ref_len / hyp_len))

    return bp * math.exp(log_avg) * 100   # scale to 0-100

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    # TODO: Task 3 — loop test set, decode, compute and return BLEU
    model.eval()
    hypotheses = []
    references = []

    SKIP_IDS = {1, 2, 3}   # PAD, SOS, EOS

    def ids_to_str(ids):
        tokens = []
        for idx in ids:
            if idx in SKIP_IDS:
                continue
            tokens.append(tgt_vocab.get(idx, "<unk>") if isinstance(tgt_vocab, dict)
                          else tgt_vocab.lookup_token(idx))
        return " ".join(tokens)

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)

            for i in range(src.size(0)):
                src_i  = src[i:i+1]
                mask_i = make_src_mask(src_i)

                result = greedy_decode(
                    model, src_i, mask_i, max_len,
                    start_symbol=2, end_symbol=3, device=device
                )

                hypotheses.append(ids_to_str(result[0].tolist()))
                references.append(ids_to_str(tgt[i].tolist()))

    return _corpus_bleu(hypotheses, references)


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pth",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # TODO: implement using torch.save({...}, path)
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'model_config':         model.model_config,
        'src_stoi':             model.src_stoi,
        'tgt_stoi':             model.tgt_stoi,
        'src_itos':             model.src_itos,
        'tgt_itos':             model.tgt_itos,
    }, path)
    print(f"Checkpoint saved to {path} (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    print(f"Checkpoint loaded from {path} (epoch {ckpt['epoch']})")
    return ckpt['epoch']

# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # TODO: implement full experiment
    import wandb
    from dataset import Multi30kDataset, collate_fn
    from torch.utils.data import DataLoader
    from lr_scheduler import NoamScheduler

    config = {
        'd_model': 256, 'N': 4, 'num_heads': 8, 'd_ff': 512,
        'dropout': 0.1, 'warmup_steps': 4000, 'batch_size': 128,
        'num_epochs': 50, 'smoothing': 0.1,
    }
    wandb.init(project="da6401-a3", config=config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Datasets
    train_ds = Multi30kDataset('train')

    val_ds = Multi30kDataset('validation')
    val_ds.load_vocab(train_ds)
    val_ds.process_data()

    test_ds = Multi30kDataset('test')
    test_ds.load_vocab(train_ds)
    test_ds.process_data()

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'],
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=False, collate_fn=collate_fn)

    # Model (checkpoint_path=None → no download during training)
    model = Transformer(
        src_vocab_size=len(train_ds.src_vocab),
        tgt_vocab_size=len(train_ds.tgt_vocab),
        d_model=config['d_model'], N=config['N'],
        num_heads=config['num_heads'], d_ff=config['d_ff'],
        dropout=config['dropout'],
        checkpoint_path=None,          
    ).to(device)

    # Attach vocab dicts so save_checkpoint can pack them
    model.src_stoi = train_ds.src_vocab
    model.tgt_stoi = train_ds.tgt_vocab
    model.src_itos = train_ds.src_itos
    model.tgt_itos = train_ds.tgt_itos

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=config['d_model'],
                              warmup_steps=config['warmup_steps'])
    loss_fn   = LabelSmoothingLoss(len(train_ds.tgt_vocab), pad_idx=1,
                                   smoothing=config['smoothing'])

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler,
                               epoch_num=epoch, is_train=True,  device=device)
        val_loss   = run_epoch(val_loader,   model, loss_fn, None, None,
                               epoch_num=epoch, is_train=False, device=device)

        print(f"Epoch {epoch+1:02d}: train={train_loss:.4f}  val={val_loss:.4f}")
        wandb.log({'epoch': epoch, 'train_loss': train_loss,
                   'val_loss': val_loss, 'lr': optimizer.param_groups[0]['lr']})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch,
                            path="best_checkpoint.pth")
            print(f"  ✓ New best saved (val_loss={val_loss:.4f})")

    # ── Final BLEU on test set ─────────────────────────────────────────
    load_checkpoint("best_checkpoint.pth", model)
    bleu = evaluate_bleu(model, test_loader, train_ds.tgt_itos, device=device)
    print(f"Test BLEU: {bleu:.2f}")
    wandb.log({'test_bleu': bleu})
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()