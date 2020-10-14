import os
import io
from time import time

from nltk.corpus import stopwords
from nltk import word_tokenize
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.wordnet import WordNetLemmatizer

class CreateEmbeddingModels:
    """
    Description
    -----------
    Creates a Word2Vec model for each dataset passed by parameters (based on Google News Word2Vec model).

    Parameters
    ----------
    embedding_file_path: str
        The complete embedding pre-treined model file path.
    document_path: str
        The raw document file path.
    path_to_save_model: str
        Path to save the result Word2Vec model.
    embedding_type: boolean
        To specify if the pre-treined embedding model is binary.
    """

    def __init__(self, embedding_file_path, embedding_type, document_path, path_to_save_model):
        self.document_path = document_path
        self.path_to_save_model = path_to_save_model
        self._read_embedding(embedding_file_path, embedding_type)
        self._make_dir(path_to_save_model)
        self.stop_words = list(set(stopwords.words('english')))
        self.useful_words = ['not', 'never', 'less', 'without', 'cannot', 'nobody', 'none',
                        'call','after', 'while','beyond', 'several', 'again', 'front',
                        'always', 'one', 'two', 'three', 'four', 'five', 'sometime', 
                        'another', 'nor', 'on', '!','?', 'no', "jpg", 'jpeg', 'png']

    def create_embedding_models(self, dataset):
        """
        Description
        -----------
        Create the cluwords for one dataset,

        Parameters
        ----------
        dataset: str
            The name of the dataset in documents_path to create the filtered embedding model.

        Return
        ------
        n_words: int
            Number of words of dataset present in the pre-treined model.
        """
        documents = self._read_raw_dataset(self.document_path)

        # Count the words in dataset
        dataset_cv = CountVectorizer().fit(documents)
        dataset_words = dataset_cv.get_feature_names()

        # Select just the words in dataset from Google News Word2Vec Model
        words_values = []
        for i in dataset_words:
            aux = [i + ' ']
            try:
                for k in self.model[i]:
                    aux[0] += str(k) + ' '
            except KeyError:
                continue

            words_values.append(aux[0])

        # filtered_vocabulary = []
        # for wv in words_values:
        #     filtered_vocabulary.append(wv.split(' ')[0])

        n_words = len(words_values)  # Number of words selected

        print('{}:{}'.format(dataset, n_words))

        # save .txt model
        file = open("""{}/{}.txt""".format(self.path_to_save_model, dataset), 'w+', encoding="utf-8")
        file.write('{0} {1}\n'.format(n_words, '300'))
        for word_vec in words_values:
            file.write("%s\n" % word_vec)

        return n_words

    def _read_embedding(self, embedding_file_path, binary):
        t0 = time()
        self.model = KeyedVectors.load_word2vec_format(
            embedding_file_path, binary=binary)
        print('Embedding model read in %0.3fs.' % (time() - t0))

    def _read_raw_dataset(self, document_path):
        arq = open(document_path, 'r', encoding="utf-8")
        doc = arq.readlines()
        arq.close()
        documents = list(map(str.rstrip, doc))
        lemmatizer = WordNetLemmatizer()
        for i in range(len(documents)):
            word_tokens = word_tokenize(documents[i])
            # word_tokens = [w for w in word_tokens if not w in self.stop_words or w in self.useful_words]
            word_tokens = [lemmatizer.lemmatize(w) for w in word_tokens if (w not in self.stop_words or w in self.useful_words) and str.isalpha(w)]

            documents[i] = ' '.join(word_tokens)
        return documents

    def _make_dir(self, path_to_save_model):
        os.system('mkdir ' + path_to_save_model)
