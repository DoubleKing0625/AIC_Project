
import nltk
import string, pprint, os, sys, csv, logging

import numpy as np

from operator import itemgetter
from collections import Counter, OrderedDict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from gensim import corpora, models, similarities

DOC_DIR = "../data/texts"
QUES_DIR = "../data/questions"

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def get_sentences(file):
    with open(file, 'r', encoding='latin-1') as d:
        text = d.read()
        tmp = nltk.sent_tokenize(text)
        sentences = [clean_text(sent) for sent in tmp]
    return sentences

def get_tokens(file):
    with open(file, 'r', encoding='latin-1') as d:
        text = d.read()
        tokens = nltk.word_tokenize(clean_text(text))
    return tokens


def get_most_common_tokens(tokens, num):
    count = Counter(tokens)
    return count.most_common(num)


def get_pos_tag(tokens):
    tags = nltk.pos_tag(tokens)
    return tags


def get_lemma(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas


def get_stem(tokens):
    porter_stemmer = PorterStemmer()
    stems = [porter_stemmer.stem(token) for token in tokens]
    return stems


def remove_stop_word(tokens):
    normalized_tokens = [token for token in tokens if token not in stopwords.words('english')]
    normalized_sentence = " ".join(normalized_tokens)
    return normalized_sentence


def remove_stop_tag(token_tags):
    stop_tag = ['IN', 'DT', 'CC', 'TO', 'PRP', 'MD', 'WDT', 'WP', 'RP', 'EX', 'PDT', 'WP$', 'UH']
    normalized_tags = [token for token, tag in token_tags if tag not in stop_tag]
    normalized_sentence = " ".join(normalized_tags)
    return normalized_sentence


def clean_text(text):
    lowers = text.lower()  # lower case for everyone
    # remove the punctuation using the character deletion step of translate
    punct_killer = str.maketrans('', '', string.punctuation)
    no_punctuation = lowers.translate(punct_killer)
    return no_punctuation


def get_text(file):
    with open(file, 'r', encoding='latin-1') as d:
        text = d.read()
    return clean_text(text)


def tokenize(text):
    return nltk.word_tokenize(text)


def create_tfidf(dir):
    text_list, text_names = [], []
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".txt"):
                # print("treating "+file)
                file_path = subdir + os.path.sep + file
                text_list.append(get_text(file_path))
                text_names.append(file_path)

    return text_list, text_names


# def tf(tokens, token):
#     count = Counter(tokens)
#     tf = count[token] / sum(count.values())
#     return tf


def get_cosine_similarity(text1, text2, v):
    t1 = v.transform([text1])
    t2 = v.transform([text2])
    return cosine_similarity(t1, t2)
    # return entropy(t1.toarray()[0,:],t2.toarray()[0,:])


# def get_euclidean_distances(text1, text2, v):
#     t1 = v.transform([text1])
#     t2 = v.transform([text2])
#     return euclidean_distances(t1, t2)
#
#
# def get_jaccard_similarity(query, document):
#     intersection = set(query).intersection(set(document))
#     union = set(query).union(set(document))
#     return len(intersection) / len(union)

def get_similarity_gensim(model, query):
    index = similarities.MatrixSimilarity(model)
    # similarities.Similarity(model)
    # index = similarities.SparseMatrixSimilarity(model, num_features=12)
    # print(list(index)[0])
    # similarity with each text theme
    return index[query]

def get_query():
    query = input("Please input your query: ")
    return query

def get_query_gensim():
    query = input("Please input your query: ")
    query = tokenize(clean_text(query))
    query_bow = dictionary.doc2bow(query)
    return query_bow

# def expand_morpho(morphology, tokens):
#     lemmas = get_lemma(tokens)
#     lemma_morpho = {lemma: morpho for morpho in morphology for lemma in lemmas if lemma in morpho}
#     return lemma_morpho


# def expand_syno(synonym, tokens):
#     lemmas = get_lemma(tokens)
#     lemma_syno = {lemma: syno for syno in synonym for lemma in lemmas if lemma in syno}
#     return lemma_syno

# def search_reply(sent):
#     reply = ""
#     return reply

if __name__ == '__main__':
    # -----A tfidf model using sklearn----- #
    # train model in whole corpus
    text_list, text_names = create_tfidf(DOC_DIR)

    # v = TfidfVectorizer(encoding='latin-1', tokenizer=tokenize, stop_words='english')
    # tfidf = v.fit_transform(text_list)

    texts = {file: get_text(file) for file in text_names}
    sentences = get_sentences("../data/all.txt")

    eval = {}

    # while 1:
    #     query = get_query()
    #
    #     # look for reply per scene, than per sentence with fixed scene
    #     similarity_text = {text_file: get_cosine_similarity(query, txt, v)[0][0] for text_file, txt in texts.items()}
    #     sorted_similarity_text = sorted(similarity_text.items(), key=lambda x: x[1], reverse=True)
    #
    #     sentences = get_sentences(sorted_similarity_text[0][0])
    #
    #     similarity_sentences = {sent: get_cosine_similarity(query, sent, v)[0][0] for sent in sentences}
    #     sorted_similarity_sentences = sorted(similarity_sentences.items(), key=lambda x: x[1], reverse=True)
    #
    #     print("possible answers of system A are: ")
    #     print(sorted_similarity_sentences[0:5])
    #     print("the reply of system A is: " + sorted_similarity_sentences[0][0])
    #
    #     # look for reply in the whole text
    #     similarity_sentences = {sent: get_cosine_similarity(query, sent, v)[0][0] for sent in sentences}
    #     sorted_similarity_sentences = sorted(similarity_sentences.items(), key=lambda x: x[1], reverse=True)
    #
    #     print("possible answers of system B are: ")
    #     print(sorted_similarity_sentences[0:5])
    #     print("the reply of system B is: " + sorted_similarity_sentences[0][0])
    #
    #     while 1:
    #         eval_tmp = input("\nWhich system you think is better(A, B, 0, 1 where O means neither, 1 means both)? ")
    #         if eval_tmp in ['A', 'B', '0', '1']:
    #             eval[query] = eval_tmp
    #             break

        # print(eval)


    # all_stems = sum(texts_stemmed, [])
    # stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
    # texts = [[stem for stem in text if stem not in stems_once] for text in texts_stemmed]




    # -----A tfidf model using gensim----- #

    corpus = [get_tokens(file) for file in text_names]
    # print(corpus)
    dictionary = corpora.Dictionary(corpus)
    # print(dictionary)
    # print(dictionary.token2id)
    # word frequence
    doc_vectors = [dictionary.doc2bow(text) for text in corpus]
    # print(doc_vectors[0])

    tfidf = models.TfidfModel(doc_vectors)
    tfidf_vectors = tfidf[doc_vectors]
    # print(tfidf_vectors)
    # print(tfidf_vectors[0])

    # query = get_query_gensim()
    # sims_tfidf = get_similarity_gensim(tfidf_vectors, query)
    # print(len(sims_tfidf))


    # -----A lsi model using gensim----- #
    lsi = models.LsiModel(tfidf_vectors, id2word=dictionary, num_topics=2)
    lsi_vectors = lsi[tfidf_vectors]
    # for vec in lsi_vector:
    #     print(vec)

    # query_lsi = lsi[query]
    # sims_lsi = get_similarity_gensim(lsi_vectors, query_lsi)
    # print(list(enumerate(sims_lsi)))


    # -----A lda model using gensim----- #
    lda = models.LdaModel(doc_vectors, id2word=dictionary, num_topics=2)
    lda_vectors = lda[tfidf_vectors]
    print(lda.print_topics(2))

    # query_lda = lda[query]
    # sims_lda = get_similarity_gensim(lda_vectors, query_lda)

    # Random Projections
    model = models.RpModel(tfidf_vectors, num_topics=500)
    # Hierarchical Dirichlet Process
    model = models.HdpModel(doc_vectors, id2word=dictionary)




