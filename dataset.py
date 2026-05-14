import spacy
from datasets import load_dataset
from collections import Counter
import torch
from torch.utils.data import Dataset

# Special token constants
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]
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
        raw = load_dataset("bentrevett/multi30k")
        self.data = raw[split]

        # Loading spaCy tokenizers
        self.de_nlp = spacy.load("de_core_news_sm")
        self.en_nlp = spacy.load("en_core_web_sm")

        # Building vocab from training split ONLY
        train_data = raw["train"]
        self.src_vocab, self.src_itos = self.build_vocab(
            [ex["de"] for ex in train_data],
            self.de_nlp,
            min_freq
        )
        self.tgt_vocab, self.tgt_itos = self.build_vocab(
            [ex["en"] for ex in train_data],
            self.en_nlp,
            min_freq
        )

        # Tokenizing and numericalizing this split
        self.src_data, self.tgt_data = self.process_data()
    
    def tokenize(self, text, nlp):
        """Lowercase tokenize using spaCy. Returns list of strings."""
        return [tok.text.lower() for tok in nlp(text)]
    
    def build_vocab(self,sentences, nlp, min_freq=2):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent
        counter = Counter()
        for sentence in sentences:
            tokens = self.tokenize(sentence, nlp)
            counter.update(tokens)

        # Starting with special tokens
        stoi = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        itos = {i: tok for i, tok in enumerate(SPECIAL_TOKENS)}

        # Adding frequent tokens
        for token, freq in counter.items():
            if freq >= min_freq and token not in stoi:
                idx = len(stoi)
                stoi[token] = idx
                itos[idx] = token

        return stoi, itos
    
    def numericalize(self, tokens, stoi):
        """
        Convert list of token strings to list of integer indices.
        Wrap with <sos> and <eos>.
        Unknown tokens map to UNK_IDX.
        """
        ids = [SOS_IDX]
        for tok in tokens:
            ids.append(stoi.get(tok, UNK_IDX))
        ids.append(EOS_IDX)
        return ids

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices
        src_data, tgt_data = [], []
        for example in self.data:
            src_tokens = self.tokenize(example["de"], self.de_nlp)
            tgt_tokens = self.tokenize(example["en"], self.en_nlp)
            src_data.append(self.numericalize(src_tokens, self.src_vocab))
            tgt_data.append(self.numericalize(tgt_tokens, self.tgt_vocab))
        return src_data, tgt_data
    
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    batch: list of (src_ids, tgt_ids) tuples (Python lists)
    Returns: src_tensor [B, max_src_len], tgt_tensor [B, max_tgt_len]
    """
    src_batch, tgt_batch = zip(*batch)
    # Convert to tensors
    src_tensors = [torch.tensor(s, dtype=torch.long) for s in src_batch]
    tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in tgt_batch]
    # Pad to max length in batch, pad_value=PAD_IDX=1
    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded