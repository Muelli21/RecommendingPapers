import os
import gc
import json
import requests

import numpy as np
import pandas as pd

import utils.constants as constants
import preparation.extracting as extracting
from utils.text_processing import text2vec, load_embeddings, text2vec_adjusted

from sortedcontainers import SortedKeyList
from catboost import CatBoostClassifier

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

HEADERS = {}

def best_candidate_cosine_similarity(paper_id, papers, embeddings, dim, content='content'):
    """
    Returns the paper that is most similar to the paper identified by the provided 'paper_id'
    in terms of cosine distance between the paper embeddings (averaged word embeddings).

    Params:
        paper_id: hash identifying the queried paper
        embeddings: dict-like object containing word embeddings
        similarity metric: "Content", "Abstract" - specifies whether the whole content or only the abstract are used
    
    Returns:
        tuple containing: 
            1. the hash of the closest paper
            2. the distance to the closest paper
    """

    paper_vector = text2vec(papers[paper_id][content], embeddings, dim)
    paper_vector = paper_vector.reshape(1, -1)

    candidate_ids = list(papers.keys())
    candidate_ids.remove(paper_id)

    candidate_vectors = np.array([ text2vec(papers[Hash][content], embeddings, dim) for Hash in candidate_ids ])

    best_tuple = pairwise_distances_argmin_min(paper_vector, candidate_vectors, metric='cosine')
    best_argmin = best_tuple[0][0]
    best_min = 1 - best_tuple[1][0]
    best_hash = candidate_ids[best_argmin]

    return (best_hash, best_min)

def rank_candidates_cosine_similarity(paper_id, papers, top_n, embeddings, dim, content='content'):
    """
    Returns the 'top_n' papers that are most similar to the paper identified by the provided 'paper_id'
    in terms of cosine distance between the paper embeddings (averaged word embeddings).

    Params:
        paper_id: hash identifying the queried paper
        embeddings: dict-like object containing word embeddings
        similarity metric: "Content", "Abstract" - specifies whether the whole content or only the abstract are used
    
    Returns:
        tuple containing:
            1. the hash of the closest paper
            2. the distance to the closest paper
    """

    paper_vector = text2vec(papers[paper_id][content], embeddings, dim)
    paper_vector = paper_vector.reshape(1, -1)

    candidate_ids = list(papers.keys())
    candidate_ids.remove(paper_id)

    candidate_vectors = np.array([ text2vec(papers[Hash][content], embeddings, dim) for Hash in candidate_ids ])

    distances = pairwise_distances(paper_vector, candidate_vectors, metric='cosine')
    distances = distances.ravel().tolist()

    tuples = list(zip(candidate_ids, distances))
    tuples = sorted(tuples, key=lambda tup: tup[1])[:top_n]

    return tuples

def rank_candidates_cosine_similarity_adjusted(paper_id, papers, top_n, embeddings, dim, content='content'):
    """
    Returns the 'top_n' papers that are most similar to the paper identified by the provided 'paper_id'
    in terms of cosine distance between the paper embeddings (averaged word embeddings).

    Params:
        paper_id: hash identifying the queried paper
        embeddings: dict-like object containing word embeddings
        similarity metric: "Content", "Abstract" - specifies whether the whole content or only the abstract are used
    
    Returns:
        tuple containing:
            1. the hash of the closest paper
            2. the distance to the closest paper
    """

    subject_embeddings, _ = load_embeddings(constants.SUBJECT_EMBEDDINGS)
    
    paper_vector = text2vec_adjusted(papers[paper_id][content], embeddings, dim, papers[paper_id]["subject"][0], subject_embeddings)
    paper_vector = paper_vector.reshape(1, -1)

    candidate_ids = [ key for key, value in papers.items() if key != paper_id and "subjects" in value.keys() ]
    candidate_vectors = np.array([ text2vec_adjusted(papers[Hash][content], embeddings, dim, papers[Hash][content]["subject"][0], subject_embeddings) for Hash in candidate_ids ])

    distances = pairwise_distances(paper_vector, candidate_vectors, metric='cosine')
    distances = distances.ravel().tolist()

    tuples = list(zip(candidate_ids, distances))
    tuples = sorted(tuples, key=lambda tup: tup[1])[:top_n]

    return tuples

def rank_candidates_per_subject(paper, top_n, embeddings, dim, subjects, include_embeddings = True, stringify = False, keywords = None):

    paper_vector = text2vec(paper["abstract"], embeddings, dim)
    paper_vector = paper_vector.reshape(1, -1)

    top_n_by_subject = {}

    for subject in subjects:

        subject_files_path = constants.SUBJECT_FILES_DIR / subject
        top_n_by_subject[subject] = SortedKeyList([], key=lambda tup: tup[1])

        for filename in os.listdir(subject_files_path):
            if ".csv" in filename or ".parquet" in filename:

                data = pd.read_csv(str(subject_files_path / filename)) if ".csv" in filename else pd.read_parquet(str(subject_files_path / filename))

                if keywords is not None:
                    data = data[data["ABSTRACT"].apply(lambda abstract: all(k in abstract for k in keywords))]
                    # Skip data frame if no rows are left after filter
                    if len(data) == 0:
                        continue

                candidate_vectors = data["ABSTRACT"].apply(lambda text: text2vec(text, embeddings, dim)).values
                candidate_vectors = np.stack(candidate_vectors)
                candidate_ids = data["PAPER_ID"].values.astype(str) if stringify else data["PAPER_ID"].values
                candidate_titles = data["TITLE"].values.astype(str) if stringify else data["TITLE"].values
                candidate_abstracts = data["ABSTRACT"].values.astype(str) if stringify else data["ABSTRACT"].values

                distances = pairwise_distances(paper_vector, candidate_vectors, metric='cosine')
                distances = distances.ravel().tolist()

                tuples = list(zip(candidate_ids, distances, [subject] * len(candidate_ids), candidate_titles, candidate_abstracts, [filename] * len(candidate_ids), candidate_vectors)) if include_embeddings else list(zip(candidate_ids, distances, [subject] * len(candidate_ids), candidate_titles, candidate_abstracts, [filename] * len(candidate_ids),))

                top_n_by_subject[subject].update(tuples)
                top_n_by_subject[subject] = SortedKeyList(
                    top_n_by_subject[subject][:top_n], 
                    key=lambda tup: tup[1]
                )

                del data
                del distances
                del candidate_ids
                del candidate_vectors

                gc.collect()

        top_n_by_subject[subject] = top_n_by_subject[subject][:top_n]

    return top_n_by_subject

def rank_candidates_adjusted(paper, top_n, embeddings, dim, subjects, include_embeddings = True, stringify = False, keywords = None):

    abstract = paper["abstract"]

    model = CatBoostClassifier()
    model.load_model(constants.SUBJECT_LABELING_MODEL_SINGLE)
    paper_subject = model.predict([abstract])[0]

    subject_embeddings, _ = load_embeddings(constants.SUBJECT_EMBEDDINGS)
    top_n_candidates = SortedKeyList([], key=lambda tup: tup[1])

    paper_vector = text2vec_adjusted(abstract, embeddings, dim, paper_subject, subject_embeddings).reshape(1, -1)

    for subject in subjects:
        
        subject_files_path = constants.SUBJECT_FILES_DIR / subject

        for filename in os.listdir(subject_files_path):
            if ".csv" in filename or ".parquet" in filename:

                data = pd.read_csv(str(subject_files_path / filename)) if ".csv" in filename else pd.read_parquet(str(subject_files_path / filename))

                if keywords is not None:
                    data = data[data["ABSTRACT"].apply(lambda abstract: all(k in abstract for k in keywords))]
                    # Skip data frame if no rows are left after filter
                    if len(data) == 0:
                        continue

                candidate_vectors = data["ABSTRACT"].apply(lambda text: text2vec_adjusted(text, embeddings, dim, subject, subject_embeddings)).values
                candidate_vectors = np.stack(candidate_vectors)
                candidate_ids = data["PAPER_ID"].values.astype(str) if stringify else data["PAPER_ID"].values
                candidate_titles = data["TITLE"].values.astype(str) if stringify else data["TITLE"].values
                candidate_abstracts = data["ABSTRACT"].values.astype(str) if stringify else data["ABSTRACT"].values

                distances = pairwise_distances(paper_vector, candidate_vectors, metric='cosine')
                distances = distances.ravel().tolist()

                tuples = list(zip(candidate_ids, distances, [subject] * len(candidate_ids), candidate_titles, candidate_abstracts, [filename] * len(candidate_ids), candidate_vectors)) if include_embeddings else list(zip(candidate_ids, distances, [subject] * len(candidate_ids), candidate_titles, candidate_abstracts, [filename] * len(candidate_ids)))

                top_n_candidates.update(tuples)
                top_n_candidates = SortedKeyList(
                    top_n_candidates[:top_n], 
                    key=lambda tup: tup[1]
                )

                del data
                del distances
                del candidate_ids
                del candidate_vectors

                gc.collect()

    return top_n_candidates[:top_n]

# TODO: Add parent ID so that the references and citations can be displayed in a tree for visualization
def prepare_raw_paper(raw_paper, key):

    paper = raw_paper[key]
    paper_id = paper['paperId']

    paper_dict = {
        'title': paper['title'], 
        'abstract': paper['abstract'], 
        'year': paper['year'], 
        'authors': paper['authors'], 
        'referenceCount': paper['referenceCount'] if paper['referenceCount'] is not None else 0, 
        'citationCount': paper['citationCount'] if paper['citationCount'] is not None else 0,
        'fieldsOfStudy': extracting.clean_s2orc_categories(paper['fieldsOfStudy']),
    }

    return paper_id, paper_dict

def get_papers(paper_id, type, key, limit):

    references_url = "https://api.semanticscholar.org/graph/v1/paper/%s/%s?fields=title,abstract,authors,year,referenceCount,citationCount,fieldsOfStudy&limit=%d"
    references_response = requests.get(references_url % (paper_id, type, limit), headers=HEADERS)
    reference_response_decoded = references_response.content.decode('utf-8')

    if (references_response.status_code != 200): 
        print("Error:", reference_response_decoded)
        return {}

    references_raw = json.loads(reference_response_decoded)['data']
    references = dict(prepare_raw_paper(raw_paper, key) for raw_paper in references_raw if raw_paper[key]["paperId"] is not None)
    return references

def get_references(paper_id): 
    return get_papers(paper_id, "references", "citedPaper", 1000)

def get_citations(paper_id):
    return get_papers(paper_id, "citations", "citingPaper", 1000)

def build_citation_neighborhood(paper_id):

    references = get_references(paper_id)
    citations = {}

    for reference_id, reference in references.items():
        reference_citations = get_citations(reference_id)
        reference["citedByIds"] = list(reference_citations.keys())

        citations = citations | reference_citations

    return citations

def rank_citation_neighborhood(paper_id, paper, top_n, embeddings, dim, include_embeddings = True, stringify = False, keywords = None): 

    citations = build_citation_neighborhood(paper_id)

    if len(citations) == 0: 
        return []

    data = pd.DataFrame.from_dict(citations, orient="index")

    data = data.reset_index(drop=False)
    data["fieldsOfStudy"] = data["fieldsOfStudy"].apply(lambda fields: ", ".join(fields) if fields is not None else None)
    data = data.rename(columns={"index":"PAPER_ID", "title":"TITLE", "abstract":"ABSTRACT", "fieldsOfStudy":"SUBJECTS"})
    data = data[data["ABSTRACT"].notnull()]

    if keywords is not None:
        data = data[data["ABSTRACT"].apply(lambda abstract: all(k in abstract for k in keywords))]
        # Skip data frame if no rows are left after filter
        if len(data) == 0:
            return []

    paper_vector = text2vec(paper["abstract"], embeddings, dim)
    paper_vector = paper_vector.reshape(1, -1)

    candidate_vectors = data["ABSTRACT"].apply(lambda text: text2vec(text, embeddings, dim)).values
    candidate_vectors = np.stack(candidate_vectors)
    candidate_ids = data["PAPER_ID"].values.astype(str) if stringify else data["PAPER_ID"].values
    candidate_titles = data["TITLE"].values.astype(str) if stringify else data["TITLE"].values
    candidate_abstracts = data["ABSTRACT"].values.astype(str) if stringify else data["ABSTRACT"].values
    candidate_subjects = data["SUBJECTS"].values.astype(str) if stringify else data["SUBJECTS"].values

    distances = pairwise_distances(paper_vector, candidate_vectors, metric='cosine')
    distances = distances.ravel().tolist()

    tuples = list(zip(candidate_ids, distances, candidate_subjects, candidate_titles, candidate_abstracts, candidate_vectors)) if include_embeddings else list(zip(candidate_ids, distances, candidate_subjects, candidate_titles, candidate_abstracts))

    top_n_candidates = SortedKeyList(tuples, key=lambda tup: tup[1])
    return top_n_candidates[:top_n]