import json
import string
from collections import Counter

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
# --------------------------------------------------------------------------------
#
#                                    VOCABULARY
#
# --------------------------------------------------------------------------------


class Vocabulary(object):
    """Class to process text and extract vocabulary for mapping"""

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
            add_unk (bool): a flag that indicates whether to add the UNK token
            unk_token (str): the UNK token to add into the Vocabulary
        """

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def to_serializable(self):
        """ returns a dictionary that can be serialized """
        return {'token_to_idx': self._token_to_idx,
                'add_unk': self._add_unk,
                'unk_token': self._unk_token}

    @classmethod
    def from_serializable(cls, contents):
        """ instantiates the Vocabulary from a serialized dictionary """
        return cls(**contents)

    def add_token(self, token):
        """Update mapping dicts based on the token.

        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        """Add a list of tokens into the Vocabulary

        Args:
            tokens (list): a list of string tokens
        Returns:
            indices (list): a list of indices corresponding to the tokens
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """Retrieve the index associated with the token
          or the UNK index if token isn't present.

        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary)
              for the UNK functionality
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """Return the token associated with the index

        Args:
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


# --------------------------------------------------------------------------------
#
#                                    VECTORIZER
#
# --------------------------------------------------------------------------------


class NewsVectorizer:
    def __init__(self, text_vocab: Vocabulary, title_vocab: Vocabulary):

        self.text_vocab = text_vocab
        self.title_vocab = title_vocab

    def vectorize_text(self, text: str):
        one_hot = np.zeros(len(self.text_vocab), dtype=np.float32)

        for token in text.split(" "):
            if token not in string.punctuation:
                one_hot[self.text_vocab.lookup_token(token)] = 1

        return one_hot

    def vectorize_title(self, text: str):
        one_hot = np.zeros(len(self.title_vocab), dtype=np.float32)

        for token in text.split(" "):
            if token not in string.punctuation:
                one_hot[self.title_vocab.lookup_token(token)] = 1

        return one_hot

    @classmethod
    def from_dataframe(cls, news_df: pd.DataFrame, cutoff=25):
        text_vocab = Vocabulary(add_unk=True)
        title_vocab = Vocabulary(add_unk=True)

        word_counts_text = Counter()
        word_counts_title = Counter()

        def count_words(words, w_c):
            for w in words.split(" "):
                if (w not in string.punctuation) and (w not in stop_words):
                    w_c[w] += 1
            return w_c

        for text in news_df.text:
            word_counts_text = count_words(str(text), word_counts_text)
        for title in news_df.title:
            word_counts_title = count_words(str(title), word_counts_title)

        for word, count in word_counts_text.items():
            if count > cutoff:
                text_vocab.add_token(word)

        for word, count in word_counts_title.items():
            if count > cutoff:
                title_vocab.add_token(word)

        return cls(text_vocab, title_vocab)

    @classmethod
    def from_serializable(cls, contents):
        return cls(Vocabulary.from_serializable(contents['text_vocab']),
                   Vocabulary.from_serializable(contents['title_vocab']))

    def to_serializable(self):
        return {
            'text_vocab': self.text_vocab.to_serializable(),
            'title_vocab': self.title_vocab.to_serializable()
        }


# --------------------------------------------------------------------------------
#
#                                    DATASET
#
# --------------------------------------------------------------------------------


class NewsDataset(Dataset):
    def __init__(self, news_df: pd.DataFrame, vectorizer: NewsVectorizer):
        self.news_df = news_df
        self._vectorizer = vectorizer

        def _get_split(df, split='train'):
            split_df = df[df.split == split]
            split_size = len(split_df)

            return split_df, split_size

        self.train_df, self.train_size = _get_split(self.news_df, 'train')
        self.val_df, self.val_size = _get_split(self.news_df, 'val')
        self.test_df, self.test_size = _get_split(self.news_df, 'test')

        self._lookup_dict = {
            'train': (self.train_df, self.train_size),
            'val': (self.val_df, self.val_size),
            'test': (self.test_df, self.test_size)
        }

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, news_csv):
        news_df = pd.read_csv(news_csv)
        train_news_df = news_df[news_df.split == 'train']

        return cls(news_df, NewsVectorizer.from_dataframe(train_news_df))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, news_csv, vectorizer_filepath):
        news_df = pd.read_csv(news_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)

        return cls(news_df, vectorizer)

    @classmethod
    def load_vectorizer_only(cls, vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return NewsVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        with open(vectorizer_filepath, 'w') as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        return self._vectorizer

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        row = self._target_df.iloc[index]

        title_vector = self._vectorizer.vectorize_title(str(row.title))
        text_vector = self._vectorizer.vectorize_text(str(row.text))

        return {'x_data': np.concatenate([title_vector, text_vector]),
                'y_target': self._target_df.label.iloc[index]}

    def get_num_batches(self, batch_size):
        return len(self) // batch_size


def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


# --------------------------------------------------------------------------------
#
#                                    DATASET
#
# --------------------------------------------------------------------------------


class NewsDatasetWithSplit(Dataset):
    def __init__(self, news_df: pd.DataFrame, vectorizer: NewsVectorizer, split: str):
        self.news_df = news_df
        self._vectorizer = vectorizer
        self.split = split

        split_df = self.news_df[self.news_df.split == self.split]
        split_size = len(split_df)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, news_csv, split):
        news_df = pd.read_csv(news_csv)
        train_news_df = news_df[news_df.split == 'train']

        return cls(news_df, NewsVectorizer.from_dataframe(train_news_df), split)

    @classmethod
    def load_dataset_and_load_vectorizer(cls, news_csv, vectorizer_filepath, split):
        news_df = pd.read_csv(news_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)

        return cls(news_df, vectorizer)

    @classmethod
    def load_vectorizer_only(cls, vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return NewsVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        with open(vectorizer_filepath, 'w') as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        return self._vectorizer
