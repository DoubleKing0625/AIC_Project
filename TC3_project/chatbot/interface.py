#!/usr/bin/python

import postprogress as pp

if __name__ == '__main__':
    while 1:
        q = input("What do you wanna say to this stupid chatbot? ")
        query = pp.get_query_gensim(q)
        choose = input("Which modle do you choose?(A: tfdif, B: bm25, C: lsi+tfdif, D: lda+tfdif, E: doc2vec) :")

        query_bow = pp.dictionary.doc2bow(query)
        if choose == 'A':
            sims_tfidf = pp.get_similarity_gensim(pp.tfidf_vectors, len(pp.dictionary), query_bow)
            print(sims_tfidf[0:5])
            print("the reply of system A (tfidf) is: ----- " + pp.sentences_original[sims_tfidf[0][0]])
        elif choose == 'B':
            scores = pp.bm25Model.get_scores(query, pp.average_idf)
            max_score_idx = scores.index(max(scores))
            print("the reply of system B (bm25) is: ----- " + pp.sentences_original[max_score_idx])
        elif choose == 'C':
            query_lsi = pp.lsi[query_bow]
            # sorted similarity with each scene 1*115
            sims_lsi = pp.get_similarity_gensim(pp.lsi_vectors, 50, query_lsi)


            sentences_lsi = []
            sentences_lsi_original = []
            for i in range(5):
                sentences_lsi = sentences_lsi + pp.get_sentences_clean(pp.text_names[sims_lsi[i][0]])
                sentences_lsi_original = sentences_lsi_original + pp.get_sentences_original(pp.text_names[sims_lsi[i][0]])

            corpus_lsi = [pp.remove_stop_word(pp.tokenize(line), gensim=True) for line in sentences_lsi]
            dictionary_lsi = pp.corpora.Dictionary(corpus_lsi)
            doc_vectors_lsi = [dictionary_lsi.doc2bow(text) for text in corpus_lsi]

            tfidf_vectors_lsi = pp.tfidf[doc_vectors_lsi]

            query_bow_lsi = dictionary_lsi.doc2bow(query)

            sims_lsi_scene = pp.get_similarity_gensim(tfidf_vectors_lsi, len(dictionary_lsi), query_bow_lsi)
            print(sims_lsi_scene[0:5])
            print("the reply of system C (lsi) is: ----- " + sentences_lsi_original[sims_lsi_scene[0][0]])
        elif choose == 'D':
            query_lda = pp.lda[query_bow]
            sims_lda = pp.get_similarity_gensim(pp.lda_vectors, 50, query_lda)


            sentences_lda = []
            sentences_lda_original = []
            for i in range(5):
                sentences_lda = sentences_lda + pp.get_sentences_clean(pp.text_names[sims_lda[i][0]])
                sentences_lda_original = sentences_lda_original + pp.get_sentences_original(
                    pp.text_names[sims_lda[i][0]])

            corpus_lda = [pp.remove_stop_word(pp.tokenize(line), gensim=True) for line in sentences_lda]
            dictionary_lda = pp.corpora.Dictionary(corpus_lda)
            doc_vectors_lda = [dictionary_lda.doc2bow(text) for text in corpus_lda]

            tfidf_vectors_lda = pp.tfidf[doc_vectors_lda]

            query_bow_lda = dictionary_lda.doc2bow(query)

            sims_lda_scene = pp.get_similarity_gensim(tfidf_vectors_lda, len(dictionary_lda), query_bow_lda)
            print(sims_lda_scene[0:5])
            print("the reply of system D (lda) is: ----- " + sentences_lda_original[sims_lda_scene[0][0]])
        elif choose == 'E':
            sims_doc2vec = pp.get_similarity_gensim(pp.doc2vec, None, query, d2v=True)
            print(sims_doc2vec[0:5])
            print("the reply of system E (doc2vec) is: ----- " + pp.sentences_original[sims_doc2vec[0][0]])
        else:
            print("Wrong inout, restart!")
