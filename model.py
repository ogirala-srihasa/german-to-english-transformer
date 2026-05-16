"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]

# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)                                    # depth per head
    scores = Q @ K.transpose(-2, -1)                    # [..., seq_q, seq_k]
    scores = scores / math.sqrt(d_k)                    # scale

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))  # mask=True → -inf

    attn_weights = F.softmax(scores, dim=-1)            # [..., seq_q, seq_k]
    output = attn_weights @ V                           # [..., seq_q, d_v]
    return output, attn_weights


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    mask = (src == pad_idx)                  # [batch, src_len]  True=PAD
    return mask.unsqueeze(1).unsqueeze(2)    # [batch, 1, 1, src_len]


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    # tgt: [batch, tgt_len]
    B, T = tgt.shape
    
    # Padding mask
    pad_mask = (tgt == pad_idx)                           # [B, T]
    pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)         # [B, 1, 1, T]
    
    # Causal mask: upper triangle (excluding diagonal) is True (masked)
    causal_mask = torch.triu(torch.ones(T, T, device=tgt.device), diagonal=1).bool()
    # causal_mask: [T, T]
    
    # Combine: True if EITHER padding OR future position
    tgt_mask = pad_mask | causal_mask                     # broadcasts to [B, 1, T, T]
    return tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        
        # 4 linear projections: Q, K, V, and output
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]

        """
        B = query.size(0)

        # Project and reshape into heads
        def project_and_split(x, W):
            # x: [B, seq, d_model]
            return W(x).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
            # output: [B, H, seq, d_k]

        Q = project_and_split(query, self.W_Q)   # [B, H, seq_q, d_k]
        K = project_and_split(key,   self.W_K)   # [B, H, seq_k, d_k]
        V = project_and_split(value, self.W_V)   # [B, H, seq_k, d_k]

        # Attention (mask broadcasts: [B,1,1,seq_k] or [B,1,T,T] → [B,H,seq_q,seq_k])
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)
        # attn_out: [B, H, seq_q, d_k]

        # Concatenate heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        # attn_out: [B, seq_q, d_model]

        return self.W_O(attn_out)   # [B, seq_q, d_model]


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Pre-compute PE matrix: [1, max_len, d_model]
        pe  = torch.zeros(max_len, d_model)                    # [max_len, d_model]
        pos = torch.arange(0, max_len).unsqueeze(1).float()    # [max_len, 1]

        # div_term: [d_model/2]  (denominator for even indices)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div_term)   # even dims
        pe[:, 1::2] = torch.cos(pos * div_term)   # odd dims

        pe = pe.unsqueeze(0)                       # [1, max_len, d_model]
        self.register_buffer('pe', pe)             # NOT a parameter!

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]  

        """
        x = x + self.pe[:, :x.size(1), :]    # slice to actual seq length
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, max_len = 5000) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)    # expand
        self.linear2 = nn.Linear(d_ff, d_model)    # contract
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO:instantiate:
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: instantiate:
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff         = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # 1. Masked Self-Attention (Q=K=V=x, use tgt_mask for causality)
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))

        # 2. Cross-Attention (Q=x from decoder, K=V=memory from encoder)
        #    Use src_mask to ignore encoder's padding
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))

        # 3. FFN
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.d_model if hasattr(layer, 'd_model') else layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """
    

    def __init__(
        self,
        src_vocab_size: int =  7853,
        tgt_vocab_size: int =  5893,
        d_model:   int   = 256,
        N:         int   = 4,
        num_heads: int   = 8,
        d_ff:      int   = 512,
        dropout:   float = 0.1,
        checkpoint_path: str = "auto",
    ) -> None:
        super().__init__()
        # TODO: Instantiate 
        # Vocab placeholders
        self.src_stoi: dict = {}
        self.tgt_stoi: dict = {}
        self.src_itos: dict = {}
        self.tgt_itos: dict = {}

        ckpt = None
        if checkpoint_path == "auto":
            # GDRIVE_ID  = "1YIwEhGSNEWDPIAG1tUwSiWSfAqNWs5zt" 
            GDRIVE_ID  = "1P2Bm2mVHjFaVgis812TVX1HchsByiEnx" 
            local_path = "best_checkpoint.pth"
            if not os.path.exists(local_path):
                gdown.download(id=GDRIVE_ID, output=local_path, quiet=False)
            checkpoint_path = local_path

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            # Extract vocab dicts packed by save_checkpoint
            if 'src_stoi' in ckpt:
                self.src_stoi = ckpt['src_stoi']
                self.tgt_stoi = ckpt.get('tgt_stoi', {})
                self.src_itos = ckpt.get('src_itos', {})
                self.tgt_itos = ckpt.get('tgt_itos', {})
                # Override sizes so architecture matches checkpoint exactly
                src_vocab_size = len(self.src_stoi)
                tgt_vocab_size = len(self.tgt_stoi)
            if 'model_config' in ckpt:
                cfg = ckpt['model_config']
                d_model   = cfg.get('d_model',   d_model)
                N         = cfg.get('N',         N)
                num_heads = cfg.get('num_heads', num_heads)
                d_ff      = cfg.get('d_ff',      d_ff)
                dropout   = cfg.get('dropout',   dropout)

        # Build architecture
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc   = PositionalEncoding(d_model, dropout)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.fc_out  = nn.Linear(d_model, tgt_vocab_size)

        self.model_config = {
            'src_vocab_size': src_vocab_size, 'tgt_vocab_size': tgt_vocab_size,
            'd_model': d_model, 'N': N, 'num_heads': num_heads,
            'd_ff': d_ff, 'dropout': dropout,
        }
        self._init_weights()

        # Load weights after architecture is built
        if ckpt is not None and 'model_state_dict' in ckpt:
            self.load_state_dict(ckpt['model_state_dict'])

        # spaCy tokenizer for infer()
        import spacy, subprocess, sys
        try:
            self.de_nlp = spacy.load("de_core_news_sm")
        except OSError:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "de_core_news_sm"])
            self.de_nlp = spacy.load("de_core_news_sm")
        

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
    
        x = self.pos_enc(self.src_embed(src))    # [B, src_len, d_model]
        return self.encoder(x, src_mask)          # [B, src_len, d_model]

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        x = self.pos_enc(self.tgt_embed(tgt))               # [B, tgt_len, d_model]
        x = self.decoder(x, memory, src_mask, tgt_mask)     # [B, tgt_len, d_model]
        return self.fc_out(x)                               # [B, tgt_len, tgt_vocab_size]

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)


    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.
        
        Args:
            src_sentence: The raw German text.
            
            
        Returns:
            The fully translated English string, detokenized and clean.
        """
        self.eval()
        device = next(self.parameters()).device

        # Tokenize German input
        tokens  = [tok.text.lower() for tok in self.de_nlp.tokenizer(src_sentence)]
        src_ids = [SOS_IDX] + [self.src_stoi.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]
        src     = torch.tensor([src_ids], dtype=torch.long, device=device)  # [1, src_len]

        #  Encode
        src_mask = make_src_mask(src)
        with torch.no_grad():
            memory = self.encode(src, src_mask)

        # Greedy decode
        ys = torch.tensor([[SOS_IDX]], device=device)
        for _ in range(100):
            tgt_mask = make_tgt_mask(ys)
            with torch.no_grad():
                logits     = self.decode(memory, src_mask, ys, tgt_mask)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == EOS_IDX:
                break

        # Convert indices to English string
        out_tokens = [
            self.tgt_itos.get(idx, "<unk>")
            for idx in ys[0].tolist()
            if idx not in (SOS_IDX, EOS_IDX, PAD_IDX)
        ]
        return " ".join(out_tokens)