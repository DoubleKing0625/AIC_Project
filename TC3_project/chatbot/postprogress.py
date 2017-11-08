import nltk
import string, pprint, os, sys, csv, logging, multiprocessing, itertools

import numpy

from operator import itemgetter
from collections import Counter, OrderedDict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from gensim import corpora, models, similarities
from gensim.summarization import bm25

DOC_DIR = "../data/texts"
QUES_DIR = "../data/questions"

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_sentences_original(file):
    with open(file, 'r', encoding='latin-1') as d:
        text = d.read()
        tmp = nltk.sent_tokenize(text)
        #sentences = [clean_text(sent) for sent in tmp]
    return tmp

def get_sentences_clean(file):
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


def remove_stop_word(tokens, gensim=False):
    normalized_tokens = [token for token in tokens if token not in stopwords.words('english')]
    if gensim:
        return normalized_tokens
    else:
        normalized_sentence = " ".join(normalized_tokens)
        return normalized_sentence


def remove_stop_tag(token_tags, gensim=False):
    stop_tag = ['IN', 'DT', 'CC', 'TO', 'PRP', 'MD', 'WDT', 'WP', 'RP', 'EX', 'PDT', 'WP$', 'UH']
    normalized_tags = [token for token, tag in token_tags if tag not in stop_tag]
    if gensim:
        return normalized_tags
    else:
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

def get_similarity_gensim(model, num_feature, query, d2v=False):
    if d2v:
        return doc2vec.docvecs.most_similar([doc2vec.infer_vector(query)], topn=len(doc2vec.docvecs))
    else:
        index = similarities.MatrixSimilarity(model, num_features=num_feature)
        sims = index[query]
        sims = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        return sims


def get_query():
    query = input("Please input your query: ")
    query = remove_stop_word(tokenize(clean_text(query)))
    return query


def get_query_gensim(query):
    query = remove_stop_word(tokenize(clean_text(query)), gensim=True)
    return query



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


# -----A tfidf model using sklearn----- #
text_list, text_names = create_tfidf(DOC_DIR)
sentences = get_sentences_clean("../data/all.txt")
sentences_original = get_sentences_original("../data/all.txt")

texts = {file: get_text(file) for file in text_names}


# -----A tfidf model using gensim----- #
corpus = [remove_stop_word(tokenize(line), gensim=True) for line in sentences]
dictionary = corpora.Dictionary(corpus)
# word frequence tf
doc_vectors = [dictionary.doc2bow(text) for text in corpus]


tfidf = models.TfidfModel(doc_vectors)
tfidf_vectors = tfidf[doc_vectors]


# -----A bm25 model using gensim----- #
bm25Model = bm25.BM25(corpus)
average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())

# -----A lsi model using gensim----- #
corpus_scene = [remove_stop_word(tokenize(line), gensim=True) for line in text_list]
dictionary_scene = corpora.Dictionary(corpus_scene)
doc_vectors_scene = [dictionary_scene.doc2bow(text) for text in corpus_scene]

tfidf_scene = models.TfidfModel(doc_vectors_scene)
tfidf_vectors_scene = tfidf_scene[doc_vectors_scene]

lsi = models.LsiModel(tfidf_vectors_scene, id2word=dictionary_scene, num_topics=30)
# topic weighted num_document*num_topics
lsi_vectors = lsi[tfidf_vectors_scene]

# -----A lda model using gensim----- #
lda = models.LdaModel(doc_vectors_scene, id2word=dictionary_scene, num_topics=30, iterations=3000)
lda_vectors = lda[doc_vectors_scene]


# -----A doc2vec model using gensim----- #
corpus_doc2vec = [models.doc2vec.TaggedDocument(remove_stop_word(tokenize(line), gensim=True), [i]) for i, line in enumerate(sentences)]

doc2vec = models.Doc2Vec(size=24, min_count=2, workers=multiprocessing.cpu_count())
# Build a Vocabulary
doc2vec.build_vocab(corpus_doc2vec)


