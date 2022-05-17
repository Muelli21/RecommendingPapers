import os
import numpy as np
import pandas as pd
import multiprocessing
import utils.constants as constants
from functools import partial
from utils.text_processing import text2vec, load_embeddings, text2vec_adjusted
from sortedcontainers import SortedKeyList
from catboost import CatBoostClassifier
from sklearn.metrics.pairwise import pairwise_distances

def rank_candidates_per_subject(paper, top_n, embeddings, dim, subjects, include_embeddings = True, stringify = False):

    paper_vector = text2vec(paper["abstract"], embeddings, dim)
    paper_vector = paper_vector.reshape(1, -1)

    pool = multiprocessing.Pool(processes = 4)
    
    partial_function = partial(subject_ranking, 
        paper_vector = paper_vector, 
        top_n = top_n, 
        embeddings = embeddings, 
        dim = dim, 
        include_embeddings = include_embeddings, 
        stringify = stringify
    )

    top_n_by_subject = pool.map(partial_function, subjects)

    return dict(top_n_by_subject)

def subject_ranking(subject, paper_vector, top_n, embeddings, dim, include_embeddings, stringify): 

    subject_files_path = constants.EMBEDDED_SUBJECT_FILES_DIR / subject
    sorted_list = SortedKeyList([], key=lambda tup: tup[1])

    for filename in os.listdir(subject_files_path):
        if ".csv" in filename or ".parquet" in filename:

            data = pd.read_csv(str(subject_files_path + filename)) if ".csv" in filename else pd.read_parquet(str(subject_files_path + filename))

            candidate_vectors = data["EMBEDDING"].values
            candidate_vectors = np.stack(candidate_vectors)
            candidate_ids = data["PAPER_ID"].values.astype(str) if stringify else data["PAPER_ID"].values
            candidate_titles = data["TITLE"].values.astype(str) if stringify else data["TITLE"].values
            candidate_abstracts = data["ABSTRACT"].values.astype(str) if stringify else data["ABSTRACT"].values

            distances = pairwise_distances(paper_vector, candidate_vectors, metric='cosine')
            distances = distances.ravel().tolist()

            tuples = list(zip(candidate_ids, distances, [subject] * len(candidate_ids), candidate_titles, candidate_abstracts, [filename] * len(candidate_ids), candidate_vectors)) if include_embeddings else list(zip(candidate_ids, distances, [subject] * len(candidate_ids), candidate_titles, candidate_abstracts, [filename] * len(candidate_ids),))

            sorted_list.update(tuples)
            sorted_list = SortedKeyList(
                sorted_list[:top_n], 
                key=lambda tup: tup[1]
            )

    return (subject, sorted_list[:top_n])

def rank_candidates_adjusted(paper, top_n, embeddings, dim, subjects, include_embeddings = True, stringify = False):

    abstract = paper["abstract"]

    model = CatBoostClassifier()
    model.load_model(constants.SUBJECT_LABELING_MODEL_SINGLE)
    paper_subject = model.predict([abstract])[0]

    subject_embeddings, _ = load_embeddings(constants.SUBJECT_EMBEDDINGS)
    paper_vector = text2vec_adjusted(abstract, embeddings, dim, paper_subject, subject_embeddings).reshape(1, -1)

    pool = multiprocessing.Pool(processes = 4)
    
    partial_function = partial(adjusted_ranking, 
        paper_vector = paper_vector, 
        top_n = top_n, 
        embeddings = embeddings, 
        dim = dim, 
        subject_embeddings = subject_embeddings,
        include_embeddings = include_embeddings, 
        stringify = stringify
    )

    sorted_lists = pool.map(partial_function, subjects)
    sorted_lists_flattened = [entry for sorted_list in sorted_lists for entry in sorted_list]

    return SortedKeyList(sorted_lists_flattened, key=lambda tup: tup[2])[:top_n]

def adjusted_ranking(subject, paper_vector, top_n, embeddings, dim, subject_embeddings, include_embeddings, stringify): 

    subject_files_path = constants.SUBJECT_FILES_DIR / subject
    sorted_list = SortedKeyList([], key=lambda tup: tup[1])

    for filename in os.listdir(subject_files_path):
        if ".csv" in filename or ".parquet" in filename:

            data = pd.read_csv(subject_files_path + filename) if ".csv" in filename else pd.read_parquet(subject_files_path + filename)

            candidate_vectors = data["ABSTRACT"].apply(lambda text: text2vec_adjusted(text, embeddings, dim, subject, subject_embeddings)).values
            candidate_vectors = np.stack(candidate_vectors)
            candidate_ids = data["PAPER_ID"].values.astype(str) if stringify else data["PAPER_ID"].values
            candidate_titles = data["TITLE"].values.astype(str) if stringify else data["TITLE"].values
            candidate_abstracts = data["ABSTRACT"].values.astype(str) if stringify else data["ABSTRACT"].values

            distances = pairwise_distances(paper_vector, candidate_vectors, metric='cosine')
            distances = distances.ravel().tolist()

            tuples = list(zip(candidate_ids, distances, [subject] * len(candidate_ids), candidate_titles, candidate_abstracts, [filename] * len(candidate_ids), candidate_vectors)) if include_embeddings else list(zip(candidate_ids, distances, [subject] * len(candidate_ids), candidate_titles, candidate_abstracts, [filename] * len(candidate_ids)))

            sorted_list.update(tuples)
            sorted_list = SortedKeyList(
                sorted_list[:top_n], 
                key=lambda tup: tup[1]
            )

    return sorted_list[:top_n]