from flask import Flask, render_template, request
import pandas as pd
import pickle
import string

from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)

# =========================================
# LOAD DATA
# =========================================

PICKLE_DIR = "pickle"
CORPUS_DIR = "corpus"

with open(f"{PICKLE_DIR}/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(f"{PICKLE_DIR}/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

with open(f"{PICKLE_DIR}/thesaurus.pkl", "rb") as f:
    thesaurus = pickle.load(f)

paper_x = pd.read_csv(
    f"{CORPUS_DIR}/hasil_crawling.csv"
)

paper_x["Kalimat"] = paper_x["Kalimat"].fillna("")

# =========================================
# PREPROCESS
# =========================================

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

stemmer = StemmerFactory().create_stemmer()

remove_punctuation_map = str.maketrans(
    '',
    '',
    string.punctuation
)

# =========================================
# QUERY EXPANSION
# =========================================

def expand_query(query):

    query = query.lower()

    query = query.translate(
        remove_punctuation_map
    )

    query = stopword.remove(query)

    tokens = query.split()

    tokens = [
        stemmer.stem(t)
        for t in tokens
    ]

    expanded = []

    for token in tokens:

        if token in thesaurus:

            expanded += thesaurus[token]

        else:

            expanded.append(token)

    return ' '.join(set(expanded))

# =========================================
# SEARCH ENGINE
# =========================================

def search(query, top_n=50):

    expanded_query = expand_query(query)

    query_vec = vectorizer.transform(
        [expanded_query]
    )

    similarities = cosine_similarity(
        query_vec,
        tfidf_matrix
    ).flatten()

    # ambil ranking similarity terbesar
    top_idx = similarities.argsort()[::-1][:top_n]

    results = paper_x.iloc[top_idx].copy()

    results["Score"] = similarities[top_idx]

    # filter dokumen relevan
    results = results[
        results["Score"] > 0.001
    ]

    # sorting descending
    results = results.sort_values(
        by="Score",
        ascending=False
    )

    return results[
        [
            "Judul",
            "Tanggal",
            "URL",
            "Score"
        ]
    ]

# =========================================
# ROUTE
# =========================================

@app.route("/", methods=["GET", "POST"])

def home():

    results = None
    total_results = 0
    query = ""

    if request.method == "POST":

        query = request.form["query"]

        data = search(
            query=query,
            top_n=50
        )

        total_results = len(data)

        results = data.to_dict(
            "records"
        )

    return render_template(
        "index.html",
        results=results,
        total_results=total_results,
        query=query
    )

# =========================================
# RUN FLASK
# =========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )

