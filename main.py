import os
from optparse import OptionParser
from script_functions import create_embedding_models
from script_functions import generate_topics
from script_functions import save_cluword_representation

from datetime import datetime

def main():
    parser = OptionParser(usage="usage: %prog [options] corpus_file")
    parser.add_option("-d", "--dataset", action="store", type="string", dest="dataset",
                      help="base output directory (default is current directory)", default=None)
    options, args = parser.parse_args()
    # Paths and files paths
    MAIN_INPUT_PATH = './datasets'
    MAIN_OUTPUT_PATH = '/tmp/cluhtm/{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    EMBEDDING_RESULTS = 'fasttext_wiki'
    PATH_TO_SAVE_RESULTS = '{}/{}/results'.format(MAIN_OUTPUT_PATH, EMBEDDING_RESULTS)
    PATH_TO_SAVE_MODEL = '{}/{}/datasets/gn_w2v_models'.format(MAIN_OUTPUT_PATH, EMBEDDING_RESULTS)
    DATASETS_PATH = '{}/textual_folds/{}Pre.txt'.format(MAIN_INPUT_PATH, options.dataset)
    EMBEDDINGS_FILE_PATH = '{}/wiki-news-300d-1M.vec'.format(MAIN_INPUT_PATH)
    EMBEDDINGS_BIN_TYPE = False
    DATASET = options.dataset
    CLASS_PATH = '{}/textual_folds/{}Class.txt'.format(MAIN_INPUT_PATH, options.dataset)
    N_THREADS = 6
    ALGORITHM_TYPE = 'knn_cosine'

    try:
        os.mkdir('{}'.format("/tmp/cluhtm"))
    except FileExistsError:
        pass
    
    try:
        os.mkdir('{}'.format(MAIN_OUTPUT_PATH))
        os.mkdir('{}/{}'.format(MAIN_OUTPUT_PATH, EMBEDDING_RESULTS))
        os.mkdir('{}/{}/results'.format(MAIN_OUTPUT_PATH, EMBEDDING_RESULTS))
        os.mkdir('{}/{}/datasets'.format(MAIN_OUTPUT_PATH, EMBEDDING_RESULTS))
        os.mkdir('{}/{}/datasets/gn_w2v_models'.format(MAIN_OUTPUT_PATH, EMBEDDING_RESULTS))
    except FileExistsError:
        pass

    # Create the word2vec models for each dataset
    print('Filter embedding space to {} dataset...'.format(DATASET))
    n_words = create_embedding_models(dataset=DATASET,
                                      embedding_file_path=EMBEDDINGS_FILE_PATH,
                                      embedding_type=EMBEDDINGS_BIN_TYPE,
                                      datasets_path=DATASETS_PATH,
                                      path_to_save_model=PATH_TO_SAVE_MODEL)

    print('Build topics...')
    generate_topics(dataset=DATASET,
                    word_count=n_words,
                    path_to_save_model=PATH_TO_SAVE_MODEL,
                    datasets_path=DATASETS_PATH,
                    path_to_save_results=PATH_TO_SAVE_RESULTS,
                    n_threads=N_THREADS,
                    algorithm_type=ALGORITHM_TYPE,
                    k=500,
                    threshold=0.4,
                    class_path=CLASS_PATH)

    save_cluword_representation(dataset=DATASET,
                                word_count=n_words,
                                path_to_save_model=PATH_TO_SAVE_MODEL,
                                datasets_path=DATASETS_PATH,
                                path_to_save_results=PATH_TO_SAVE_RESULTS,
                                n_threads=N_THREADS,
                                algorithm_type=ALGORITHM_TYPE,
                                k=500,
                                threshold=0.4,
                                class_path=CLASS_PATH)


if __name__ == '__main__':
    main()
