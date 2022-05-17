import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

import utils.files as files
import utils.constants as constants
import utils.text_processing as text_processing

s2orc_labels = {
    "Art": "Arts and Humanities",
    "Materials Science": "Material Sciences", 
    "Medicine": "Medical Science",
    "Sociology": "Social Sciences"
}

def update_info(counter, info_path):
    if not os.path.exists(info_path): 
        counts = pd.DataFrame(counter.items(), columns = ["SUBJECT", "COUNT"])
        counts.to_csv(info_path, index = False)
    else:
        counts = pd.read_csv(info_path)
        counts["COUNT"] = counts["COUNT"] + np.array(list(counter.values()))
        counts.to_csv(info_path, index = False)

def clean_subjects(subjects): 
    return [s2orc_labels[subject] if subject in s2orc_labels else subject for subject in subjects ]

def prepare_subject(subject): 
    return subject.replace(" ", "_").lower()

def setup_s2orc_subject_files(source_path, subject_files_path, file_format = ".csv"):

    counter = Counter(constants.SUBJECTS)

    for subject in constants.SUBJECTS:
        os.makedirs(os.path.dirname(subject_files_path / subject), exist_ok=True)

    for filename in tqdm(os.listdir(source_path)):
        if ".gz" in filename:
            print("Processing " + filename)

            for paper in files.load_compressed_jsonl(str(source_path / filename)):

                abstract = paper["abstract"]
                subjects = paper["mag_field_of_study"]

                if abstract is not None and abstract != "NA" and subjects is not None:
                    
                    subjects = clean_subjects(subjects)

                    paper_id = paper["paper_id"]
                    title = paper["title"].replace('"', "'")
                    abstract = abstract.replace('"', "'")
                    subject = subjects[0]
                    
                    prepared_subject = prepare_subject(subject)
                    index_current_batch = int(counter[subject] / constants.BATCH_SIZE)

                    batch_name = str(subject_files_path / subject / (prepared_subject + "_" + str(index_current_batch) + file_format))
                    csv_string = paper_id + ',"' + title + '","' + abstract + '"\n'

                    # NOTE: It might make sense to store the subjects in the csv file to access all of them

                    if not os.path.exists(batch_name):
                        with open(batch_name, 'w+') as target_file:
                            target_file.write("PAPER_ID,TITLE,ABSTRACT\n")
                            target_file.write(csv_string)
                    else: 
                        with open(batch_name, 'a+') as target_file:
                            target_file.write(csv_string)

                    counter.update({subject: 1})

            # NOTE: Remove this return statement to process all files
            update_info(counter, constants.SUBJECT_INFO)
            return

def setup_merged_subject_files(source_path, subject_files_path, file_format = ".csv"):

    counter = Counter(constants.SUBJECTS)

    for subject in constants.SUBJECTS:
        os.makedirs(os.path.dirname(subject_files_path / subject), exist_ok=True)

    data = pd.read_csv(source_path) if ".csv" in source_path else pd.read_parquet(source_path)

    for paper in tqdm(data.itertuples()):

        paper_id = str(paper[0])
        title = str(paper[1]).replace('"', "'")
        abstract = text_processing.remove_whitespace(paper[2].replace('"', "'"))

        indices = np.array(paper[5:], dtype = bool)
        subjects = np.array(constants.SUBJECTS)[indices]

        csv_string = paper_id + ',"' + title + '","' + abstract + '"\n'

        # NOTE: Here, redundancy is introduced. This might be resolved by using a classifier 
        # to generate subjects instead of the assinged labels
        for subject in subjects:
            
            prepared_subject = prepare_subject(subject)
            index_current_batch = int(counter[subject] / constants.BATCH_SIZE)
            batch_name = str(subject_files_path / subject / (prepared_subject + "_merged_" + str(index_current_batch) + file_format))

            # NOTE: It might make sense to store the subjects in the csv file to access all of them

            counter.update({subject: 1})

            if not os.path.exists(batch_name):
                with open(batch_name, 'w+') as target_file:
                    target_file.write("PAPER_ID,TITLE,ABSTRACT\n")
                    target_file.write(csv_string)
            else: 
                with open(batch_name, 'a+') as target_file:
                    target_file.write(csv_string)

    update_info(counter, constants.SUBJECT_INFO)

def setup_subject_files(file_format):
    setup_s2orc_subject_files(constants.SOURCE_PATH_S2ORC, constants.SUBJECT_FILES_DIR, file_format)
    setup_merged_subject_files(constants.SOURCE_PATH_MERGED, constants.SUBJECT_FILES_DIR, file_format)

def embed_subject_files(embeddings, dim):

    for subject in constants.SUBJECTS:
        os.makedirs(os.path.dirname(constants.EMBEDDED_SUBJECT_FILES_DIR / subject), exist_ok=True)

    for subject in tqdm(constants.SUBJECTS):
        subject_files_path = constants.SUBJECT_FILES_DIR / subject
        embedded_subject_files_path = constants.EMBEDDED_SUBJECT_FILES_DIR / subject

        for filename in os.listdir(subject_files_path):
            if ".csv" in filename:
                data = pd.read_csv(str(subject_files_path / filename))
                data["EMBEDDING"] = data["ABSTRACT"].apply(lambda text: text_processing.text2vec(text, embeddings, dim))

                filename = filename.replace(".csv", ".parquet")
                data.to_parquet(str(embedded_subject_files_path / filename), index = False)

def extract_stratified_sample(n_per_subject = 100_000):

    data = pd.DataFrame()

    for subject in constants.SUBJECTS:

        subject_files_path = constants.SUBJECT_FILES_DIR / subject
        merged_data = pd.DataFrame()

        for filename in os.listdir(subject_files_path):
            if ".csv" in filename or ".parquet" in filename:
                if "merged" in filename or "0" in filename:
                    current_data = pd.read_csv(str(subject_files_path + filename)) if ".csv" in filename else pd.read_parquet(str(subject_files_path / filename))
                    merged_data = pd.concat([merged_data, current_data], axis = 0)

        merged_size = merged_data.shape[0]  if merged_data.shape[0] < n_per_subject else n_per_subject
        merged_data = merged_data.sample(n=merged_size)
        merged_data["SUBJECT"] = subject
        data = pd.concat([data, merged_data], axis = 0)

    return data

def update_subject_embeddings(embeddings, dim):
    """
    Computes and updates the average subject embeddings based on the s2orc data set
    The embeddings are stored as a .tsv-file

    Params: 
        embeddings: word embeddings used to embed papers and subsequently subjects
        dim: dimension of the word embeddings
    """

    subject_embeddings = []

    for subject in tqdm(constants.SUBJECTS):

        print("Computing embedding for", subject)

        subject_files_path = constants.SUBJECT_FILES_DIR / subject
        subject_texts = []

        for filename in os.listdir(subject_files_path):
            if ".csv" in filename or ".parquet" in filename:

                data = pd.read_csv(str(subject_files_path / filename)) if ".csv" in filename else pd.read_parquet(str(subject_files_path / filename))
                data = data.dropna(subset=["ABSTRACT"])
                data = data[data["ABSTRACT"].str.len() > 100]
                subject_texts.extend(data["ABSTRACT"].tolist())

        print("-> creating embedding based on", len(subject_texts), "papers")
        subject_embeddings.append(text_processing.text_collection2vec(subject_texts, embeddings, dim))

    embedding_frame = pd.concat([pd.DataFrame(constants.SUBJECTS), pd.DataFrame(subject_embeddings)], axis = 1)
    embedding_frame.to_csv(constants.SUBJECT_EMBEDDINGS, sep='\t', quoting=csv.QUOTE_NONE, header = False, index = False)


