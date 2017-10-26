
import nltk
import string, pprint, os, sys, csv

import numpy as np

from operator import itemgetter
from collections import Counter, OrderedDict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


DOC_DIR = "../data/texts"
QUES_DIR = "../data/questions"

def get_sentences(file):
    with open(file, 'r') as d:
        text = d.read()
        tmp = nltk.sent_tokenize(text)
        sentences = [clean_text(sent) for sent in tmp]
    return sentences

def get_tokens(file):
    with open(file, 'r') as d:
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


def tf(tokens, token):
    count = Counter(tokens)
    tf = count[token] / sum(count.values())
    return tf


def get_cosine_similarity(text1, text2, v):
    t1 = v.transform([text1])
    t2 = v.transform([text2])
    return cosine_similarity(t1, t2)
    # return entropy(t1.toarray()[0,:],t2.toarray()[0,:])


def get_euclidean_distances(text1, text2, v):
    t1 = v.transform([text1])
    t2 = v.transform([text2])
    return euclidean_distances(t1, t2)


def get_jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union)

def get_question():
    question = input("Please input your question: ")
    return question


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
    text_list, text_names = create_tfidf(DOC_DIR)

    # print("Comparing the frequency of terms in the text %s" % text_names[0] + " and their tfidf")
    # tokens_one_file = get_tokens(text_names[0])
    #
    # tfs = {token: tf(tokens_one_file, token) for token in tokens_one_file}
    # sorted_tfs = sorted(tfs.items(), key=lambda x: x[1], reverse=True)
    # for token, tf in sorted_tfs[:10]:
    #     print("\tWord: {}, TF: {}".format(token, round(tf, 5)))
    # # print sorted_tfs
    #
    # print("\n")
    #
    v = TfidfVectorizer(encoding='latin-1', tokenizer=tokenize, stop_words='english')
    tfidf = v.fit_transform(text_list)

    texts = {file: get_text(file) for file in text_names}
    while 1:
        ques = get_question()

        similarity_text = {}
        for text_file, txt in texts.items():
            similarity_text[text_file] = get_cosine_similarity(ques, txt, v)[0][0]

        sorted_similarity_text = sorted(similarity_text.items(), key=lambda x: x[1], reverse=True)
        sentences = get_sentences(sorted_similarity_text[0][0])

        similarity_sentences = {}
        for sent in sentences:
            similarity_sentences[sent] = get_cosine_similarity(ques, sent, v)[0][0]

        sorted_similarity_sentences = sorted(similarity_sentences.items(), key=lambda x: x[1], reverse=True)
        print(sorted_similarity_sentences[0][0])

        # rank = map(itemgetter(1), sorted_similarity).index(max(similarity.values()))

        # tfidf = {token: v.transform([tmp_ques])[0, int(v.vocabulary_[token])] if token in v.vocabulary_ else 0 for token in tokenize(tmp_ques)}
        # sorted_tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)

        # print("max similarity in file %s " % ques_file + " is %f " % max(similarity.values()) + "at rank %d " % (rank + 1))
        # print("the responsible descriptor is %s" % map(itemgetter(0), sorted_tfidf[:5]))


    #
    # tfidfs = {token: tfidf[text_names.index(text_names[0]), int(v.vocabulary_[token])] if token in v.vocabulary_ else 0
    #           for token in tokens_one_file}
    # sorted_tfidfs = sorted(tfidfs.items(), key=lambda x: x[1], reverse=True)
    # for token, tfidf in sorted_tfidfs[:10]:
    #     print("\tWord: {}, TF-IDF: {}".format(token, round(tfidf, 5)))
    # # print sorted_tfidfs



    # morphology = []
    # with open('data/der-families-en.utf8') as csvfile:
    #     f = csv.reader(csvfile, delimiter='\\')
    #     for row in f:
    #         morphology.append(row[0:-1:2])
    #
    # synonym = []
    # with open('data/sem-families-wn.utf8') as csvfile:
    #     f = csv.reader(csvfile, delimiter='\\')
    #     for row in f:
    #         synonym.append(row[0:-1:2])
    #
    # question_list, question_names = create_tfidf(QUES_DIR)
    #
    # # compute MRR and similarity for document 10
    # i = 10
    # questions = {file: get_text(file).strip() for file in question_names if 'rd_' + str(i) + '_' in file}
    # texts = {file: get_text(file) for file in text_names if 'rd_' + str(i) + '_' in file}
    #
    # for ques_file, ques in questions.items():
    #     # # remove stop word from request
    #     tmp_ques = remove_stop_word(tokenize(ques))
    #     # # remove unimportant tag from request
    #     tmp_ques = remove_stop_tag(get_pos_tag(tokenize(tmp_ques)))
    #     # # using stem for request
    #     # tmp_ques = " ".join(get_stem(tokenize(ques)))
    #
    #     # # using morphology for request
    #     # lemma_morpho = expand_morpho(morphology, tokenize(ques))
    #     # tmp = [" ".join(morpho) for morpho in lemma_morpho.values()]
    #     # expand = " ".join(tmp)
    #     # expand = clean_text(expand)
    #     # tmp_ques = ques + " " + expand
    #
    #     # # using synonym for request
    #     # lemma_syno = expand_syno(synonym, tokenize(ques))
    #     # tmp = [" ".join(syno) for syno in lemma_syno.values()]
    #     # expand = " ".join(tmp)
    #     # expand = clean_text(expand)
    #     # tmp_ques = ques + " " + expand
    #
    #     # using request itself
    #     # tmp_ques = ques
    #     # print tmp_ques
    #
    #     similarity = {}
    #     for text_file, txt in texts.items():
    #         # apply stem to document
    #         # tmp_txt = " ".join(get_stem(tokenize(txt)))
    #         # print tmp_txt
    #
    #         # different similrity mesure
    #         similarity[(ques_file, text_file)] = get_cosine_similarity(tmp_ques, txt, v)[0][0]
    #         # similarity[(ques_file, text_file)] = get_jaccard_similarity(tmp_ques, txt)
    #         # similarity[(ques_file, text_file)] = get_euclidean_distances(tmp_ques, txt, v)
    #
    #     sorted_similarity = sorted(similarity.items(), key=lambda x: x[1], reverse=True)
    #     rank = map(itemgetter(1), sorted_similarity).index(max(similarity.values()))
    #
    #     tfidf = {token: v.transform([tmp_ques])[0, int(v.vocabulary_[token])] if token in v.vocabulary_ else 0 for token
    #              in tokenize(tmp_ques)}
    #     sorted_tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
    #
    #     print
    #     "max similarity in file %s " % ques_file + " is %f " % max(similarity.values()) + "at rank %d " % (rank + 1)
    #     print
    #     "the responsible descriptor is %s" % map(itemgetter(0), sorted_tfidf[:5])
    #
    #     # # relevance feedback
    #
    #     # max_document = sorted_similarity[rank][0][1]
    #     # tfidf_document = {token: v.transform([texts.get(max_document)])[0, int(v.vocabulary_[token])] if token in v.vocabulary_ else 0 for token in tokenize(texts.get(max_document))}
    #     # sorted_tfidf_document = sorted(tfidf_document.items(), key = lambda x: x[1], reverse = True)
    #
    #     # expand = " ".join(OrderedDict(sorted_tfidf_document).keys()[0:5])
    #     # tmp_ques = ques + " " + expand
    #
    #     # similarity = {}
    #     # for text_file, txt in texts.items():
    #     #     similarity[(ques_file, text_file)] = get_cosine_similarity(tmp_ques, txt, v)[0][0]
    #
    #     # sorted_similarity = sorted(similarity.items(), key = lambda x: x[1], reverse = True)
    #     # rank = map(itemgetter(1), sorted_similarity).index(max(similarity.values()))
    #
    #     # tfidf = {token: v.transform([tmp_ques])[0, int(v.vocabulary_[token])] if token in v.vocabulary_ else 0 for token in tokenize(tmp_ques)}
    #     # sorted_tfidf = sorted(tfidf.items(), key = lambda x: x[1], reverse = True)
    #
    #     # print "relevance feedback : max similarity in file %s " %ques_file + " is %f " %max(similarity.values()) + "at rank %d " %(rank+1)
    #     # print "relevance feedback : the responsible descriptor is %s" %map(itemgetter(0), sorted_tfidf[:5])
    #
    #     f = open(os.path.join('MRR', os.path.basename(ques_file)), 'w')
    #     f.writelines(
    #         file[0] + '\t' + os.path.basename(file[1]) + '\t' + str(simi) + '\n' for file, simi in sorted_similarity)

