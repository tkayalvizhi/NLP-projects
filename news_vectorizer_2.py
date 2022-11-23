from news_vocabulary_vectorizer_dataset import Vocabulary, NewsVectorizer


class NewsVectorizer2(NewsVectorizer):
    def __init__(self, text_vocab: Vocabulary, title_vocab: Vocabulary):
        super(NewsVectorizer2, self).__init__(text_vocab, title_vocab)
