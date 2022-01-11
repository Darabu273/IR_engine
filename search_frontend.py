from flask import Flask, request, jsonify
import math
from collections import Counter
import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
# from google.colab import auth
from google.cloud import storage
import inverted_index_gcp
import nltk
from inverted_index_gcp import InvertedIndex

nltk.download('stopwords')
from nltk.corpus import stopwords
import re

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(stopwords.words('english'))
TUPLE_SIZE = 6  # We're going to pack the doc_id and tf values in this
# many bytes.
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

bucket_name = '314047101_316294032'  # TODO check with talya
client = storage.Client('my-project-1996-337120')
bucket = client.bucket(bucket_name)


def getIndex(bucket, indexName):
    blob = storage.Blob(f'postings_gcp/{indexName}.pkl', bucket)
    with open(f'./{indexName}.pkl', "wb") as f:
        blob.download_to_file(f)
    return inverted_index_gcp.InvertedIndex.read_index('./', indexName)


def tokenize(text):
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in stopwords_frozen]
    return list_of_tokens


def get_content_from_storage(bucket, file_name):
    blob = storage.Blob(f'{file_name}', bucket)
    with open(f'./{file_name}', "wb") as file_obj:
        blob.download_to_file(file_obj)
    with open(f'./{file_name}', 'rb') as f:
        return pickle.load(f)


'--------------Create inverted index instances for - text, title, anchor_text---------------'
print("start index text")
inverted_text = getIndex(bucket, "index_text")
inverted_title = getIndex(bucket, "index_title")
inverted_anchor = getIndex(bucket, "index_anchor")

print("finish")
'-------------- get content from storage ---------------------------------------------------'
dic_docs_pageRank = get_content_from_storage(bucket, "page_rank.pckl")
dic_docs_pageView = get_content_from_storage(bucket, "pageviews.pkl")

'-------------------------------------------------------------------------------------------'


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    token_q = tokenize(query)
    words_title, pls_title = zip(*inverted_title.posting_lists_iter(token_q))
    sorted_candidate_doc = get_candidate_documents_binaryScores(token_q, inverted_title, words_title, pls_title)
    for doc_id, tf_term in sorted_candidate_doc:
        res.append((int(doc_id), inverted_text.doc_id_title[doc_id]))
    # END SOLUTION
    return jsonify(res)

    ''' get topN for query base inverted_title'''
    # res = []
    # #query = request.args.get('query', '')
    # if len(query) == 0:
    #     return jsonify(res)
    # # BEGIN SOLUTION
    # token_q = tokenize(query)
    #
    # words_title, pls_title = zip(*inverted_text.posting_lists_iter(token_q))
    # topN_docs_query = get_topN_score_for_query(token_q, inverted_title, words_title, pls_title)
    # for doc_id, score in topN_docs_query:
    #      res.append((int(doc_id), inverted_text.doc_id_title[doc_id]))
    #  # END SOLUTION
    # return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    token_q = tokenize(query)
    words_text, pls_text = zip(*inverted_text.posting_lists_iter(token_q))
    topN_docs_query = get_topN_score_for_query(token_q, inverted_text, words_text, pls_text)
    for doc_id, score in topN_docs_query:
        res.append((int(doc_id), inverted_text.doc_id_title[doc_id]))
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    token_q = tokenize(query)
    words_title, pls_title = zip(*inverted_title.posting_lists_iter(token_q))
    sorted_candidate_doc = get_candidate_documents_binaryScores(token_q, inverted_title, words_title, pls_title)
    for doc_id, tf_term in sorted_candidate_doc:
        res.append((int(doc_id), inverted_text.doc_id_title[doc_id]))
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    token_q = tokenize(query)
    words_anchor, pls_anchor = zip(*inverted_anchor.posting_lists_iter(token_q))
    sorted_candidate_doc = get_candidate_documents_binaryScores(token_q, inverted_title, words_anchor, pls_anchor)
    for doc_id, tf_term in sorted_candidate_doc:
        res.append((int(doc_id), inverted_anchor.doc_id_title.get(doc_id, [])))

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for wiki_id in wiki_ids:
        res.append(float(dic_docs_pageRank.get(wiki_id, 0)))  # pageRank float score
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''

    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for wiki_id in wiki_ids:
        res.append(dic_docs_pageView.get(wiki_id, 0))  # pageView -ask naama if it for August 2021
    # END SOLUTION
    return jsonify(res)


'-------------- helper functions --------------'
'for search_body():'


def get_topN_score_for_query(query, index, words, pls, N=100):
    """
    Generate  for a query its topN score.
    Parameters:
    -----------
    query: a list of tokens.
    index: inverted index loaded from the all corpus.
    N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default N = 100, for the topN function.
    Returns:
    -----------
    return: a list of pairs in the following format: (doc_id, score).
    """
    quer = generate_query_tfidf_vector(query, index)  # vectorized query with tfidf scores
    doc = generate_document_tfidf_matrix(query, index, words, pls)  # DataFrame of tfidf scores.
    dic_cosim = cosine_similarity(doc, quer)
    # print(topN_docs_query)
    return get_top_n(dic_cosim, N)  # a ranked list of pairs (doc_id, score) in the length of N.


def generate_query_tfidf_vector(query_to_search, index):
    """
    Generate a vector representing the query.
    Parameters:
    -----------
    query_to_search: list of tokens (str).

    index: inverted index loaded from the all corpus.
    Returns:
    -----------
    vectorized query with tfidf scores
    """
    epsilon = .0000001
    unique = np.unique(query_to_search)
    Q = np.zeros(len(unique))  # will be the size of the query only!
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        # for token in np.array(query_to_search):
        if token in index.df.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divided by the length of the query
            df = index.df[token]
            idf = math.log((len(index.DL)) / (df + epsilon), 10)  # smoothing
            try:
                ind = unique.index(token)
                Q[ind] = tf * idf
            except:
                pass
    return Q


def generate_document_tfidf_matrix(query_to_search, index, words, pls):
    """
    Generate a DataFrame `D` of tfidf scores for a given query.
    Rows will be the documents candidates for a given query
    Columns will be the unique terms in the index.
    The value for a given document and term will be its tfidf score.

    Parameters:
    -----------
    query_to_search: list of tokens (str).
    index: inverted index loaded from the all corpus.
    words,pls: generator for working with posting.
    Returns:
    -----------
    return: DataFrame of tfidf scores.
    """
    dic_query = {}
    dic_query = dict.fromkeys(query_to_search, "0")
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words,
                                                           pls)  # We will utilize only the documents which have corresponding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), len(query_to_search)))
    D = pd.DataFrame(D)
    D.index = unique_candidates
    D.columns = dic_query.keys()
    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf
    return D


def get_candidate_documents_and_scores(query, inverted_index, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query.
    Parameters:
    -----------
    query: list of tokens (str).
    inverted_index:inverted index loaded from the all corpus.

    words,pls: generator for working with posting.
    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    N = len(inverted_index.DL)  # is the total number of documents in the collection
    for term in np.unique(query):
        if term in words:
            list_of_doc = pls[words.index(term)]
            normlized_tfidf = []
            for doc_id, freq in list_of_doc:
                if (inverted_index.DL[doc_id] == 0):
                    normlized_tfidf.append((doc_id, 0))
                else:
                    normlized_tfidf.append(
                        (doc_id, (freq / inverted_index.DL[doc_id]) * math.log(N / inverted_index.df[term], 10)))

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


def cosine_similarity(D, Q):
    """
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.
    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarity score.
    """
    # YOUR CODE HERE
    dic_cosim = {}
    for index, doc in D.iterrows():
        m1 = doc.dot(Q)
        m2 = np.linalg.norm(doc, axis=0)
        m3 = np.linalg.norm(Q)
        similarity_scores = m1 / (m2 * m3)
        dic_cosim[index] = similarity_scores

    return dic_cosim


def get_top_n(sim_dict, N=100):
    """
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    return sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]


'for search_title & search_anchor:'


def get_candidate_documents_binaryScores(query, inverted_index, words, pls):
    """
    List of candidates. In the following format: pair (doc_id, binary_score)
    """
    tf_dic = {}
    N = len(inverted_index.DL)  # is the total number of documents in the collection
    for term in np.unique(query):
        if term in words:
            list_of_doc = pls[words.index(term)]
            for doc_id, tf_term in list_of_doc:
                tf_dic[doc_id] = tf_dic.get(doc_id, 0) + tf_term

    sorted_candidate_doc = sorted([(doc_id, tf_term) for doc_id, tf_term in tf_dic.items()], key=lambda x: x[1],
                                  reverse=True)
    return sorted_candidate_doc


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
