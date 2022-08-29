import torch
import random
import numpy as np
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
from collections import namedtuple
from comp9312.ner.data import Token


def conll_to_segments(filename):
    """
    Convert CoNLL files to segments. This return list of segments and each segment is
    a list of tuples (token, tag)
    :param filename: Path
    :return: list[[tuple]] - [[(token, tag), (token, tag), ...], [(token, tag), ...]]
    """
    segments, segment = list(), list()

    with open(filename, "r") as fh:
        for token in fh.read().splitlines():
            if not token:
                segments.append(segment)
                segment = list()
            else:
                parts = token.split()
                token = Token(text=parts[0], gold_tag=parts[1])
                segment.append(token)

        segments.append(segment)

    return segments


def parse_conll_files(data_paths):
    """
    Parse CoNLL formatted files and return list of segments for each file and index
    the vocabs and tags across all data_paths
    :param data_paths: tuple(Path) - tuple of filenames
    :return: tuple( [[(token, tag), ...], [(token, tag), ...]], -> segments for data_paths[i]
                    [[(token, tag), ...], [(token, tag), ...]], -> segments for data_paths[i+1],
                    ...
                  )
             List of segments for each dataset and each segment has list of (tokens, tags)
    """
    vocabs = namedtuple("Vocab", ["tags", "tokens"])
    datasets, tags, tokens = list(), list(), list()

    for data_path in data_paths:
        dataset = conll_to_segments(data_path)
        datasets.append(dataset)
        tokens += [token.text for segment in dataset for token in segment]
        tags += [token.gold_tag for segment in dataset for token in segment]

    # Generate vocabs for tags and tokens
    counter = Counter(tags)
    counter = OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
    tag_vocab = vocab(counter, specials=["<PAD>"])

    counter = Counter(tokens)
    counter = OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
    token_vocab = vocab(counter)

    vocabs = vocabs(tokens=token_vocab, tags=tag_vocab)
    return tuple(datasets), vocabs


def set_seed(seed):
    """
    Set the seed for random intialization and set
    CUDANN parameters to ensure determmihstic results across
    multiple runs with the same seed

    :param seed: int
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
