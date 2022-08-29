from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from transformers import BertTokenizer
from functools import partial
import logging

logger = logging.getLogger(__name__)


class Token:
    def __init__(self, text=None, pred_tag=None, gold_tag=None):
        """
        Token object to hold token attributes
        :param text: str
        :param pred_tag: str
        :param gold_tag: str
        """
        self.text = text
        self.gold_tag = gold_tag
        self.pred_tag = pred_tag


class BertSeqTransform:
    def __init__(self, bert_model, vocab, max_seq_len=512):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.encoder = partial(
            self.tokenizer.encode,
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=False
        )
        self.max_seq_len = max_seq_len
        self.vocab = vocab

    def __call__(self, segment):
        subwords, tags, tokens = list(), list(), list()
        unk_token = Token(text="<UNK>")

        for token in segment:
            token_subwords = self.encoder(token.text)
            subwords += token_subwords
            tags += [self.vocab.tags[token.gold_tag]] + [self.vocab.tags["O"]] * (len(token_subwords) - 1)
            tokens += [token] + [unk_token] * (len(token_subwords) - 1)

        # Truncate to max_seq_len
        if len(subwords) > self.max_seq_len - 2:
            text = " ".join([t.text for t in tokens if t.text != "UNK"])
            logger.info("Truncating the sequence %s to %d", text, self.max_seq_len - 2)
            subwords = subwords[:self.max_seq_len - 2]
            tags = tags[:self.max_seq_len - 2]
            tokens = tokens[:self.max_seq_len - 2]

        subwords.insert(0, self.tokenizer.cls_token_id)
        subwords.append(self.tokenizer.sep_token_id)

        tags.insert(0, self.vocab.tags["O"])
        tags.append(self.vocab.tags["O"])

        tokens.insert(0, unk_token)
        tokens.append(unk_token)

        return torch.LongTensor(subwords), torch.LongTensor(tags), tokens, len(tokens)


class DefaultDataset(Dataset):
    def __init__(
            self,
            examples=None,
            vocab=None,
            bert_model="aubmindlab/bert-base-arabertv2",
            max_seq_len=512,
    ):
        """
        The dataset that used to transform the segments into training data
        :param examples: list[[tuple]] - [[(token, tag), (token, tag), ...], [(token, tag), ...]]
                         You can get generate examples from -- arabiner.data.dataset.parse_conll_files
        :param vocab: vocab object containing indexed tags and tokens
        :param bert_model: str - BERT model
        :param: int - maximum sequence length
        """
        self.transform = BertSeqTransform(bert_model, vocab, max_seq_len=max_seq_len)
        self.examples = examples
        self.vocab = vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        subwords, tags, tokens, valid_len = self.transform(self.examples[item])
        return subwords, tags, tokens, valid_len

    def collate_fn(self, batch):
        """
        Collate function that is called when the batch is called by the trainer
        :param batch: Dataloader batch
        :return: Same output as the __getitem__ function
        """
        subwords, tags, tokens, valid_len = zip(*batch)

        # Pad sequences in this batch
        # subwords and tokens are padded with zeros
        # tags are padding with the index of the O tag
        subwords = pad_sequence(subwords, batch_first=True, padding_value=0)
        tags = pad_sequence(
            tags, batch_first=True, padding_value=self.vocab.tags.get_stoi()["<PAD>"]
        )
        return subwords, tags, tokens, valid_len
