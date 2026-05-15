import sys
import subprocess
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# Special token constants
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]

def _load_spacy():
    """Load spaCy models, auto-downloading if missing."""
    try:
        import spacy
        de_nlp = spacy.load("de_core_news_sm")
        en_nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("[dataset] spaCy models not found — downloading ...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "de_core_news_sm"])
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        import spacy
        de_nlp = spacy.load("de_core_news_sm")
        en_nlp = spacy.load("en_core_web_sm")
    return de_nlp, en_nlp
 
 
def _batch_tokenize(texts: List[str], nlp, batch_size: int = 512) -> List[List[str]]:
    """
    Tokenize a list of strings using spaCy's TOKENIZER ONLY (no full pipeline).
    Uses nlp.pipe() for fast batched processing.
 
    Returns a list of token-string lists, all lowercased.
    """
    # nlp.tokenizer.pipe is the fastest path — skips POS/NER/dep-parse
    return [
        [tok.text.lower() for tok in doc]
        for doc in nlp.tokenizer.pipe(texts, batch_size=batch_size)
    ]
 
 
class Multi30kDataset:
    def __init__(self, split='train', min_freq = 2):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
        # Loading HuggingFace dataset
        self.min_freq = min_freq
 
        # Vocab structures (str→int and int→str)
        # src_vocab / tgt_vocab are stoi dicts so len() == vocab_size
        self.src_vocab: Dict[str, int] = {}   # stoi
        self.tgt_vocab: Dict[str, int] = {}   # stoi
        self.src_itos:  Dict[int, str] = {}
        self.tgt_itos:  Dict[int, str] = {}
 
        # Processed integer tensors
        self.data: List[Tuple[torch.Tensor, torch.Tensor]] = []
 
        # Loading raw HuggingFace dataset
        from datasets import load_dataset
        print(f"[dataset] Loading Multi30k — split='{split}' ...")
        raw = load_dataset("bentrevett/multi30k", split=split)
        self._raw = raw
 
        # Loading spaCy 
        self.de_nlp, self.en_nlp = _load_spacy()
 
        # Tokenising ONCE and cache
        print(f"[dataset] Batch-tokenising {len(raw)} pairs ...")
        de_texts = [ex["de"] for ex in raw]
        en_texts = [ex["en"] for ex in raw]
        self.src_tokens: List[List[str]] = _batch_tokenize(de_texts, self.de_nlp)
        self.tgt_tokens: List[List[str]] = _batch_tokenize(en_texts, self.en_nlp)
        print("[dataset] Tokenisation complete.")

        if split == 'train':
            self.build_vocab()
            self.process_data()
    
    
    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent
        if self.split != 'train':
            warnings.warn(
                f"build_vocab() called on split='{self.split}'. "
                "Build vocab on 'train' only, then use load_vocab() for other splits.",
                stacklevel=2,
            )
 
        print(f"[dataset] Building vocabularies (min_freq={self.min_freq}) ...")
 
        src_freq = Counter(tok for sent in self.src_tokens for tok in sent)
        tgt_freq = Counter(tok for sent in self.tgt_tokens for tok in sent)
 
        # Sorting for deterministic ordering across runs
        src_words = sorted(w for w, c in src_freq.items() if c >= self.min_freq)
        tgt_words = sorted(w for w, c in tgt_freq.items() if c >= self.min_freq)
 
        # Building stoi dicts (special tokens first, indices 0-3)
        self.src_vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS + src_words)}
        self.tgt_vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS + tgt_words)}
 
        # Building itos dicts
        self.src_itos = {i: tok for tok, i in self.src_vocab.items()}
        self.tgt_itos = {i: tok for tok, i in self.tgt_vocab.items()}
 
        print(
            f"[dataset] DE vocab: {len(self.src_vocab):,} tokens | "
            f"EN vocab: {len(self.tgt_vocab):,} tokens"
        )

    def load_vocab(self, train_dataset: "Multi30kDataset"):
        """
        Convert list of token strings to list of integer indices.
        Wrap with <sos> and <eos>.
        Unknown tokens map to UNK_IDX.
        """
        self.src_vocab = train_dataset.src_vocab
        self.tgt_vocab = train_dataset.tgt_vocab
        self.src_itos  = train_dataset.src_itos
        self.tgt_itos  = train_dataset.tgt_itos
        print(
            f"[dataset] Vocab injected from train split — "
            f"DE: {len(self.src_vocab):,} | EN: {len(self.tgt_vocab):,}"
        )
    

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices
        if not self.src_vocab or not self.tgt_vocab:
            raise RuntimeError(
                "Vocabulary is empty. Call build_vocab() (train) or "
                "load_vocab(train_ds) (val/test) before process_data()."
            )
 
        print(f"[dataset] Numericalising {len(self.src_tokens)} pairs ...")
        self.data = []
 
        for src_toks, tgt_toks in zip(self.src_tokens, self.tgt_tokens):
            src_ids = (
                [SOS_IDX]
                + [self.src_vocab.get(t, UNK_IDX) for t in src_toks]
                + [EOS_IDX]
            )
            tgt_ids = (
                [SOS_IDX]
                + [self.tgt_vocab.get(t, UNK_IDX) for t in tgt_toks]
                + [EOS_IDX]
            )
            self.data.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
            ))
 
        print(f"[dataset] {len(self.data):,} examples ready.")
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx: int):
        return self.data[idx]
    
    def get_dataloader(self, batch_size = 128,shuffle = True, num_workers: int  = 0):
        """Convenience wrapper — returns a ready-to-use DataLoader."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

def collate_fn(batch):
    """
    batch: list of (src_ids, tgt_ids) tuples (Python lists)
    Returns: src_tensor [B, max_src_len], tgt_tensor [B, max_tgt_len]
    """
    src_batch, tgt_batch = zip(*batch)
    # Pad to max length in batch, pad_value=PAD_IDX=1
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded