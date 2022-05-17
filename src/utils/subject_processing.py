import csv
import numpy as np
import pandas as pd
import utils.files as files
import utils.constants as constants
import preparation.database as database
from catboost import CatBoostClassifier, Pool
from utils.text_processing import text2vec, load_embeddings

# NOTE: This function only houses legacy code

def most_likely_subjects(proba):
    """
    Extracts the most likely subject/s for each paper from a 
    probability distribution over all subjects

    Params: 
        proba: 2d matrix whose rows are probability distributions over the subjects

    Returns: 
        list of the predicted subjects per paper
    """

    nr_samples = proba.shape[0]
    indices = np.argmax(proba, axis = 1)
    subjects = np.array(constants.SUBJECTS)[indices]

    # NOTE: We will have to see whether it might be enough to remove the most important average subject embedding
    # NOTE: The subjects should be ordered by importance
    return subjects.reshape((nr_samples, -1)).tolist()

def build_subject_prediction_df(papers, content):
    """
    Builds a pandas data frame consisting of paper hashes and the respective paper content
    The data frame can be used to predict subjects for each of the papers in one batch

    Params: 
        papers: papers to include in the batch
        content: text to base the prediction on

    Returns: 
        prediction_df: data frame (columns: [["hash"], ["content"]])
    """

    contents = [ (Hash, paper[content]) for Hash, paper in papers.items() if paper[content] is not None ]
    return pd.DataFrame(contents, columns = ["hash", "content"])

def predict_all_subjects(prediction_df):
    """
    Applies catboost classifier to the data frame prepared for subject prediction

    Params:
        prediction_df: data frame (columns: [["hash"], ["content"]])
    """

    # NOTE: These embeddings have to match those used for training catboost
    
    embeddings, dim = load_embeddings(constants.STARSPACE_EMBEDDINGS)
    embedded = prediction_df["content"].apply(lambda text: text2vec(text, embeddings, dim))

    pool = Pool(data = embedded)

    model = CatBoostClassifier()
    model.load_model(constants.SUBJECT_LABELING_MODEL_MULTI)
    
    probas = model.predict_proba(pool)
    most_likely = most_likely_subjects(probas)
    
    prediction_df["subjects"] = pd.Series(most_likely)

def get_papers_by_subject(subject):
    """
    Filters all papers for those belonging to the given subject

    Params: 
        subject: name of subject to embed

    Returns: 
        dict mapping hashes to papers belonging to the subject
    """

    subject_hashes = files.load_json(constants.SUBJECT_HASHES)

    if subject not in subject_hashes: 
        raise NameError("The subject you are querying does not exist!")

    single_subject_hashes = subject_hashes[subject]
    all_papers = files.load_json(constants.PAPERS)

    return {Hash: all_papers[Hash] for Hash in single_subject_hashes}

def subject2vec(subject, embeddings, dim, content):
    """
    Computes the average paper embedding of a subject

    Params: 
        subject: name of subject to embed
        embeddings: dict mapping words to vectors
        dim: dimension of word embeddings
        content: field of a paper to be used for the embedding

    Returns: 
        mean word embedding all documents of the given subject
    """

    subject_papers = get_papers_by_subject(subject)
    subject_texts = [ paper[content] for paper in subject_papers.values() ]
    subject_average = database.subject_texts2vec(subject_texts, embeddings, dim)

    return subject_average

def update_paper_subjects(papers, content):
    """
    Updates the subjects of all papers in PAPERS and updates the subjects in SUBJECT_HASHES
    NOTE: If the papers are loaded somewhere else while this method is performed and saved afterwards, the changes will be overwritten
    NOTE: SUBJECT_HASHES is overwritten, when this method is performed. I.e. it should always be performed on all papers
    """

    subject_hashes = { subject: [] for subject in constants.SUBJECTS }

    prediction_df = build_subject_prediction_df(papers, content)
    predict_all_subjects(prediction_df)

    for paper in prediction_df.itertuples(name = "Paper"):

        paper_hash = paper[1]
        paper_subjects = paper[3]

        papers[paper_hash]["subjects"] = paper_subjects

        for subject in paper_subjects: 
            subject_hashes[subject].append(paper_hash)

    files.save_json(constants.SUBJECT_HASHES, subject_hashes)

# NOTE: It might make sense to compute the subject embeddings based on the abstracts in the big datasets
# to maintain functionality even with a low number of downloaded papers
def update_subject_embeddings(embeddings, dim, content):
    """
    Computes and updates the average subject embeddings. 
    The embeddings are stored as a .tsv-file

    Params: 
        embeddings: word embeddings used to embed papers and subsequently subjects
        dim: dimension of the word embeddings
    """

    subjects_frame = pd.DataFrame(constants.SUBJECTS)

    embeddings_frame = pd.DataFrame([ subject2vec(subject, embeddings, dim, content) for subject in constants.SUBJECTS ])
    embedding_frame = pd.concat([subjects_frame, embeddings_frame], axis = 1)

    embedding_frame.to_csv(constants.SUBJECT_EMBEDDINGS,sep='\t', quoting=csv.QUOTE_NONE, header = False, index = False)