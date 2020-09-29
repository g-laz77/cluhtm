# Packages
import os
import json
import pandas as pd
import numpy as np
import logging as log
import glob
import operator
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from cluwords import Cluwords, CluwordsTFIDF
from metrics import Evaluation
from embedding import CreateEmbeddingModels
from generate_nmf import GenerateNFM
from reference_nmf import ReferenceNFM
from topic_stability import TopicStability
from text.util import save_corpus, load_corpus
from collections import deque
from os.path import join
import spacy
import re

nlp = spacy.load('en_core_web_lg', parse=True, tag=True, entity=True)


def top_words(model, feature_names, n_top_words):
    topico = []
    for topic_idx, topic in enumerate(model.components_):
        top = ''
        top2 = ''
        top += ' '.join([feature_names[i]
                         for i in topic.argsort()[:-n_top_words - 1:-1]])
        top2 += ''.join(str(sorted(topic)[:-n_top_words - 1:-1]))

        topico.append(str(top))

    return topico


def print_results(model, tfidf_feature_names, cluwords_freq, cluwords_docs,
                  dataset, path_to_save_results, path_to_save_model, sufix):
    print(path_to_save_results)
    for t in [5, 10, 20]:
        with open('{}/{}_result_topic_{}.txt'.format(path_to_save_results, sufix, t), 'w', encoding="utf-8") as f_res:
            f_res.write('Topics {}\n'.format(t))
            topics = top_words(model, tfidf_feature_names, t)
            f_res.write('{}\n'.format(topics))
            log.info('Coherence Metric...')
            coherence = Evaluation.coherence(topics, cluwords_freq, cluwords_docs)
            f_res.write('Coherence: {} ({})\n'.format(np.round(np.mean(coherence), 4), np.round(np.std(coherence), 4)))
            f_res.write('{}\n'.format(coherence))
            log.info('PMI Metric...')
            for word, freq in cluwords_freq.items():
                log.info('w {} f {}'.format(word, freq))

            pmi, npmi = Evaluation.pmi(topics, cluwords_freq, cluwords_docs,
                                       sum([freq for word, freq in cluwords_freq.items()]), t)
            f_res.write('PMI: {} ({})\n'.format(np.round(np.mean(pmi), 4), np.round(np.std(pmi), 4)))
            f_res.write('{}\n'.format(pmi))
            f_res.write('NPMI: {} ({})\n'.format(np.round(np.mean(npmi), 4), np.round(np.std(npmi), 4)))
            f_res.write('{}\n'.format(npmi))
            log.info('W2V-L1 Metric...')
            w2v_l1 = Evaluation.w2v_metric(topics, t, path_to_save_model, 'l1_dist', dataset)
            f_res.write('W2V-L1: {} ({})\n'.format(np.round(np.mean(w2v_l1), 4), np.round(np.std(w2v_l1), 4)))
            f_res.write('{}\n'.format(w2v_l1))

            f_res.close()


def save_results(model, tfidf_feature_names, path_to_save_model, dataset, cluwords_freq,
                 cluwords_docs, path_to_save_results):
    res_mean = []
    coherence_mean = ['coherence']
    lcp_mean = ['lcp']
    npmi_mean = ['npmi']
    w2v_l1_mean = ['w2v-l1']

    for t in [5, 10, 20]:
        topics = top_words(model, tfidf_feature_names, t)

        # Write topics in a file
        file = open('{}/topics_{}.txt'.format(path_to_save_results, t), 'w+', encoding="utf-8")
        file.write('TOPICS WITH {} WORDS\n\n'.format(t))
        for i, topic in enumerate(topics):
            file.write('Topic %d\n' % i)
            file.write('%s\n' % topic)
        file.close()

        coherence = Evaluation.coherence(topics, cluwords_freq, cluwords_docs)
        coherence_mean.extend(['{0:.3f} +- {1:.3f}'.format(np.mean(coherence),
                                                           np.std(coherence))])

        lcp = Evaluation.lcp(topics, cluwords_freq, cluwords_docs)
        lcp_mean.extend(['{0:.3f} +- {1:.3f}'.format(np.mean(lcp),
                                                     np.std(lcp))])

        pmi, npmi = Evaluation.pmi(topics, cluwords_freq, cluwords_docs,
                                 sum([freq for word, freq in cluwords_freq.items()]), t)
        npmi_mean.extend(['{0:.3f} +- {1:.3f}'.format(np.mean(npmi),
                                                      np.std(npmi))])

        w2v_l1 = Evaluation.w2v_metric(topics, t, path_to_save_model, 'l1_dist', dataset)
        w2v_l1_mean.extend(['{0:.3f} +- {1:.3f}'.format(np.mean(w2v_l1),
                                                        np.std(w2v_l1))])
    res_mean.extend([coherence_mean, lcp_mean, npmi_mean, w2v_l1_mean])

    df_mean = pd.DataFrame(res_mean, columns=['metric', '5 words', '10 words', '20 words'])

    df_mean.to_csv(path_or_buf='{}/results.csv'.format(path_to_save_results))


def create_embedding_models(dataset, embedding_file_path, embedding_type, datasets_path, path_to_save_model):
    # Create the word2vec models for each dataset
    word2vec_models = CreateEmbeddingModels(embedding_file_path=embedding_file_path,
                                            embedding_type=embedding_type,
                                            document_path=datasets_path,
                                            path_to_save_model=path_to_save_model)
    n_words = word2vec_models.create_embedding_models(dataset)

    return n_words


def set_cluwords_representation(dataset, out_prefix, X, class_path, path_to_save_results, datasets_path):
    loaded = np.load('{}/{}/cluwords_{}.npz'.format(path_to_save_results, "cluwords",dataset))
    terms = loaded['cluwords']
    del loaded
    y = []
    with open(class_path, 'r', encoding="utf-8") as filename:
        for line in filename:
            y.append(line.strip())

    y = np.array(y, dtype=np.int32)
    doc_ids = []
    classes = {}
    doc_id = 0
    for document_class in range(0, y.shape[0]):
        doc_ids.append(doc_id)
        if y[document_class] not in classes:
            classes[y[document_class]] = []

        classes[y[document_class]].append(doc_id)
        doc_id += 1

    save_corpus(join(path_to_save_results,"data_splits", out_prefix), X, terms, doc_ids, classes)
    with open('{}.txt'.format(join(path_to_save_results,"data_splits", out_prefix)), "w") as f:
        arq = open(datasets_path, 'r', encoding="utf-8")
        doc = arq.readlines()
        arq.close()

        documents = list(map(str.rstrip, doc))
        for id in doc_ids:
            f.write(documents[id]+"\n")
    return y

def prep_phrase_2(word, analysed_doc):
    phrase = ""
    compound = ""
    if word.pos_ in ["VERB", "NOUN", "ADJ", "PROPN"] or word.dep_ in ["xcomp", "ccomp"]:
        subtree_span = analysed_doc[word.left_edge.i : word.right_edge.i + 1]
        phrase = subtree_span.text
    return phrase.strip().lower()

def prepare_phrase(token):
    phrase = ""
    compound = ""

    if token.pos_ in ['NOUN', 'PROPN']:
        phrase = token.text
        for j in token.lefts:
            if j.dep_ == 'compound':
                compound = j.text + ' ' + token.text
            elif j.pos_ == "ADJ" and j.dep_ in ['amod','nummod']:
                str1 = j.text + ' ' + token.text
                str2 = ""
                for k in j.lefts:
                    if k.dep_ == 'advmod': 
                        str2 = k.text + ' ' + str1
                
                if str2 != "":
                    phrase = str2
                else:
                    phrase = str1
                break
          
        for l in token.rights:
            if l.pos_ == 'VERB' and l.dep_ == 'acl':
                phrase += ' ' + l.text
            elif l.dep_ == "prep":
                for k in l.rights:
                    if k.pos_ in ['NOUN', 'PROPN']:
                        phrase += ' ' + l.text + ' ' + prepare_phrase(k)
                        break
                break
            elif l.dep_ == "dobj":
                phrase += ' ' + l.text
        
        if compound != "":
            mtch = re.search(re.escape(token.text),phrase)
            if mtch is not None:
                phrase = phrase.replace(mtch.group(),compound)
    
    elif token.pos_ == "VERB":
        phrase = token.text
        for j in token.lefts:
            if (j.dep_ in ['advmod', 'neg']):
                phrase = j.text + ' ' + token.text
                break
            elif j.pos_ in ['NOUN', 'PROPN'] and j.dep_ in ['nsubj']:
                phrase = prepare_phrase(j) + ' ' + phrase
        
        for j in token.rights:
            if j.pos_ in ['NOUN', 'PROPN']:
                phrase += ' ' + prepare_phrase(j)
            elif j.dep_ == 'advmod' and j.pos_ == 'ADV':
                phrase = phrase + ' ' + j.text
            elif j.dep_ in ['dobj', 'pobj']:
                phrase = phrase + ' ' + j.text
                break
    
    elif token.pos_ == "ADJ":
        phrase = token.text
        for j,h in zip(token.rights,token.lefts):
            if j.dep_ == 'xcomp':
                for k in j.lefts:
                    if k.dep_ == 'aux':
                        phrase = h.text + ' ' + token.text + ' ' + k.text + ' ' + j.text
                        break
                break
                
    if phrase.strip() == "":
        return ""
    else:
        return phrase.strip().lower()

def fetch_topic_phrases(docs, topic_words):
    word_phrase_map = {}
    for word in topic_words:
        word_phrase_map[word] = []
        phrases = [word]

        for doc in docs:
            analysed_doc = nlp(doc)
            for token in analysed_doc:
                if token.text == word:
                    phrase = prep_phrase_2(token, analysed_doc)
                    if phrase != "" and phrase != token.text:
                        phrases.append(phrase)
        word_phrase_map[word] = list(set(phrases))

    return word_phrase_map

def save_topics(model, tfidf_feature_names, cluwords_tfidf, best_k, topics_documents, y, doc_ids, terms, out_prefix, path_to_save_results,
                dq, k_max, depth, parent, hierarchy, max_depth, datasets_path):
    topics = top_words(model, tfidf_feature_names, 10)
    for k in range(0, best_k):
        topic = np.argwhere(topics_documents == k)
        topic = topic.ravel()
        cluwords_tfidf_temp = cluwords_tfidf.copy()
        cluwords_tfidf_temp = cluwords_tfidf_temp[topic, :]

        doc_ids_temp = doc_ids.copy()
        doc_ids_temp = doc_ids_temp[topic]

        if depth not in hierarchy:
            hierarchy[depth] = {}

        if parent not in hierarchy[depth]:
            hierarchy[depth][parent] = {}

        hierarchy[depth][parent][k] = {}
        # hierarchy[depth][parent][k] = topics[k]
        classes = {}
        prefix = "{prefix} {k}".format(prefix=out_prefix, k=k)
        if len(doc_ids_temp) > k_max and depth+1 < max_depth:
        # if depth < max_depth:
            log.info("Add topic: {} Shape Matrix: {}".format(k, cluwords_tfidf_temp.shape))
            log.info("len(doc_ids): {}".format(len(doc_ids_temp)))
            for doc_id in doc_ids_temp:
                if y[doc_id] not in classes:
                    classes[y[doc_id]] = []

                classes[y[doc_id]].append(doc_id)

            # prefix = "{prefix} {k}".format(prefix=out_prefix, k=k)
            save_corpus(join(path_to_save_results,prefix), csr_matrix(cluwords_tfidf_temp), terms, doc_ids_temp, classes)

            dq.appendleft(prefix)
        # else:
        #     log.info("Exclude topic: {} Shape Matrix: {}".format(k, cluwords_tfidf_temp.shape))
        #     log.info("len(doc_ids): {}".format(len(doc_ids_temp)))
            print(topics[k].split())
        
        topic_docs = []
        with open('{}.txt'.format(join(path_to_save_results,prefix)), "w") as f:
                arq = open(datasets_path, 'r', encoding="utf-8")
                doc = arq.readlines()
                arq.close()

                documents = list(map(str.rstrip, doc))
                for id in doc_ids_temp:
                    topic_docs.append(documents[id])
                    f.write(documents[id]+"\n")
        hierarchy[depth][parent][k] = fetch_topic_phrases(topic_docs, topics[k].split())
    
    return dq, hierarchy


def print_herarchical_structure(output, hierarchy, depth=0, parent='-1', son=0):
    # print('{} {} {}'.format(depth, parent, son))
    if depth not in hierarchy:
        return

    if parent not in hierarchy[depth]:
        return

    if son not in hierarchy[depth][parent]:
        return

    tabulation = '\t' * depth
    output.write('{}{}\n'.format(tabulation, hierarchy[depth][parent][son]))
    print_herarchical_structure(output, hierarchy, depth=depth + 1, parent='{} {}'.format(parent, son),
                                son=0)
    print_herarchical_structure(output, hierarchy, depth=depth, parent=parent, son=son + 1)
    return


def generate_topics(dataset, word_count, path_to_save_model, datasets_path,
                    path_to_save_results, n_threads, k, threshold, class_path, algorithm_type, debug=3):
    log.basicConfig(filename="{}/{}.log".format(path_to_save_results, dataset), filemode="w", level=max(50 - (debug * 10), 10),
                    format='%(asctime)-18s %(levelname)-10s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d/%m/%Y %H:%M', )
    # Path to files and directories
    embedding_file_path = """{}/{}.txt""".format(path_to_save_model, dataset)
    path_to_save_results = '{}/{}'.format(path_to_save_results, dataset)
    # path_to_save_pkl = '{}/pkl'.format(path_to_save_results)

    try:
        os.mkdir('{}'.format(path_to_save_results))
    except FileExistsError:
        pass

    try:
        os.mkdir('{}/{}'.format(path_to_save_results,"data_splits"))
    except FileExistsError:
        pass
    
    try:
        os.mkdir('{}/{}'.format(path_to_save_results,"stability"))
    except FileExistsError:
        pass

    try:
        os.mkdir('{}/{}'.format(path_to_save_results,"cluwords"))
    except FileExistsError:
        pass

    Cluwords(algorithm=algorithm_type,
             embedding_file_path=embedding_file_path,
             n_words=word_count,
             k_neighbors=k,
             threshold=threshold,
             n_jobs=n_threads,
             dataset=dataset,
             path_to_save_results=join(path_to_save_results,"cluwords")
             )

    cluwords = CluwordsTFIDF(dataset=dataset,
                             dataset_file_path=datasets_path,
                             n_words=word_count,
                             path_to_save_cluwords=join(path_to_save_results,"cluwords"),
                             class_file_path=class_path)
    log.info('Computing TFIDF...')
    cluwords_tfidf = cluwords.fit_transform()
    cluwords_tfidf_temp = cluwords_tfidf.copy()
    cluwords_tfidf_temp = csr_matrix(cluwords_tfidf_temp)  # Convert the cluwords_tfidf array matrix to a sparse cluwords
    # RANGE OF TOPICS THAT WILL BE EXPLOIT BY THE STRATEGY
    k_min = 5
    k_max = 30
    n_runs = 5
    max_depth = 2
    sufix = "{dataset}_{depth}_{parent_topic}".format(dataset=dataset, depth=0, parent_topic='-1')
    y = set_cluwords_representation(dataset,
                                    sufix,
                                    cluwords_tfidf_temp,
                                    class_path,
                                    path_to_save_results, 
                                    datasets_path)
    dq = deque([sufix])
    hierarchy = {}
    while dq:
        log.info("Deque {}".format(dq))
        sufix = dq.pop()
        log.info("Starting iteration {sufix}".format(sufix=sufix))
        parent = sufix.split("_")[-1]
        depth = int(sufix.split("_")[1])
        log.info("Depth {}".format(depth))
        # if depth == max_depth:
        #     break
        log.info("Parent Topic {}".format(parent))
        log.info("Reference NMF")
        ReferenceNFM().run(dataset=dataset,
                           corpus_path="{}/{}/{}.pkl".format(path_to_save_results, "data_splits",sufix),
                           dir_out_base="{}/{}/reference-{}".format(path_to_save_results,"stability",sufix),
                           kmin=k_min,
                           kmax=k_max)
        log.info("Generate NMF")
        GenerateNFM().run(dataset=dataset,
                          corpus_path="{}/{}/{}.pkl".format(path_to_save_results,"data_splits",sufix),
                          dir_out_base="{}/{}/topic-{}".format(path_to_save_results,"stability",sufix),
                          kmin=k_min,
                          kmax=k_max,
                          runs=n_runs)
        log.info("Topic Stability")
        dict_stability = {}
        for k in range(k_min, k_max+1):
            stability = TopicStability().run(dataset=dataset,
                                             reference_rank_path="{}/{}/reference-{}/nmf_k{:02}/ranks_reference.pkl"
                                             .format(path_to_save_results,"stability",sufix, k),
                                             rank_paths=glob.glob("{}/{}/topic-{}/nmf_k{:02}/ranks*".format(path_to_save_results,"stability",sufix, k)),
                                             top=10)
            dict_stability[k] = stability

        best_k = max(dict_stability.keys(), key=(lambda key: dict_stability[key]))
        log.info("Selected K {k} => Stability({k}) = {stability} (median)".format(k=best_k,
                                                                                  stability=round(dict_stability[best_k],
                                                                                                  4)))
        X, terms, doc_ids, classes = load_corpus("{}/{}/{}.pkl".format(path_to_save_results,"data_splits",sufix))
        # Fit the NMF model
        log.info("\nFitting the NMF model (Frobenius norm) with tf-idf features, shape {}...".format(X.shape))
        nmf = NMF(n_components=best_k,
                  random_state=1,
                  alpha=.1,
                  l1_ratio=.5).fit(X)

        w = nmf.fit_transform(X)  # matrix W = m x k
        tfidf_feature_names = list(cluwords.vocab_cluwords)
        topics_documents = np.argmax(w, axis=1)

        dq, hierarchy = save_topics(model=nmf,
                                    tfidf_feature_names=tfidf_feature_names,
                                    cluwords_tfidf=X.toarray(),
                                    best_k=best_k,
                                    topics_documents=topics_documents,
                                    y=y,
                                    doc_ids=np.array(doc_ids),
                                    terms=terms,
                                    out_prefix="{dataset}_{depth}_{parent_topic}".format(dataset=dataset,
                                                                                         depth=depth+1,
                                                                                         parent_topic=parent),
                                    path_to_save_results=join(path_to_save_results,"data_splits"),
                                    dq=dq,
                                    k_max=k_max,
                                    depth=depth,
                                    parent=parent,
                                    hierarchy=hierarchy,
                                    max_depth=max_depth,
                                    datasets_path=datasets_path)
        log.info('End Iteration...')

    log.info(hierarchy)
    with open('{}/hierarchy_structure.json'.format(path_to_save_results),'w', encoding='utf-8') as f:
        json.dump(hierarchy, f)
    # output = open('{}/hierarchical_struture.txt'.format(path_to_save_results), 'w', encoding="utf-8")
    # print_herarchical_structure(output=output, hierarchy=hierarchy)
    # output.close()


def save_cluword_representation(dataset, word_count, path_to_save_model, datasets_path,
                    path_to_save_results, n_threads, k, threshold,
                    class_path, algorithm_type):
    # Path to files and directories
    embedding_file_path = """{}/{}.txt""".format(path_to_save_model, dataset)
    path_to_save_results = '{}/{}'.format(path_to_save_results, dataset)
    # path_to_save_pkl = '{}/pkl'.format(path_to_save_results)

    try:
        os.mkdir('{}'.format(path_to_save_results))
    except FileExistsError:
        pass

    Cluwords(algorithm=algorithm_type,
             embedding_file_path=embedding_file_path,
             n_words=word_count,
             k_neighbors=k,
             threshold=threshold,
             n_jobs=n_threads,
             dataset=dataset,
             path_to_save_results=join(path_to_save_results,"cluwords")
             )

    cluwords = CluwordsTFIDF(dataset=dataset,
                             dataset_file_path=datasets_path,
                             n_words=word_count,
                             path_to_save_cluwords=join(path_to_save_results,"cluwords"),
                             class_file_path=class_path)
    log.info('Computing TFIDF...')
    cluwords_tfidf = cluwords.fit_transform()
    np.savez_compressed('{}/cluwords_representation_{}.npz'.format(join(path_to_save_results,"cluwords"), dataset),
                        tfidf=cluwords_tfidf,
                        feature_names=cluwords.vocab_cluwords)
