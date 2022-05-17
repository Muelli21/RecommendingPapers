import pickle
import numpy as np
import pandas as pd

import utils.files as files
import utils.constants as constants
from utils.text_processing import text2vec, load_embeddings, prepare_string, prepare_tokens

from catboost import CatBoostClassifier, Pool
from catboost.text_processing import Dictionary
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

TOKENIZERS = [{
    'tokenizer_id': 'Space',
    'delimiter': ' ',
    'separator_type': 'ByDelimiter',
},{
    'tokenizer_id': 'Sense',
    'separator_type': 'BySense',
}]

def save_features_and_labels(data, file_path, file_name):
    embed = pd.DataFrame(np.stack(data["EMBEDDING"].values))
    embed.to_csv(str(file_path / file_name + "_features.csv"), index = False)
    data[constants.SUBJECTS].to_csv(str(file_path / file_name + "_labels.csv"), index = False)

# Multi-label Classification

def prepare_multilabel_prepared(data):

    data = data[["ABSTRACT"] + constants.SUBJECTS]
    data = data.sample(frac=1)
    data = data.reset_index(drop=True)

    embeddings, dim = load_embeddings(constants.STARSPACE_EMBEDDINGS)
    data["EMBEDDING"] = data["ABSTRACT"].apply(lambda text: text2vec(text, embeddings, dim))

    train, val = train_test_split(
        data,
        train_size=0.9,
        random_state=0
    )

    save_features_and_labels(train, constants.PROCESSED_DIR, "train")
    save_features_and_labels(val, constants.PROCESSED_DIR, "val")

def train_multilabel_prepared():

    train_pool = Pool(
        data=pd.read_csv(str(constants.PROCESSED_DIR / "train_features.csv")),
        label=pd.read_csv(str(constants.PROCESSED_DIR / "train_labels.csv")),
    )

    val_labels = pd.read_csv(str(constants.PROCESSED_DIR / "val_labels.csv"))

    val_pool = Pool(
        data=pd.read_csv(str(constants.PROCESSED_DIR / "val_features.csv")),
        label=val_labels,
    )

    model = CatBoostClassifier(
        iterations = 100,
        loss_function='MultiLogloss',
        eval_metric='Accuracy',
        class_names=val_labels.columns.values,
    )

    model.fit(train_pool, eval_set=val_pool, plot = True, use_best_model = True)
    return model

def train_multilabel(data):

    embeddings, dim = load_embeddings(constants.STARSPACE_EMBEDDINGS)

    train, val = train_test_split(
        data,
        train_size=0.8,
        random_state=0
    )

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    train_labels = train[constants.SUBJECTS]

    train_pool = Pool(
        data=train["ABSTRACT"].apply(lambda text: text2vec(text, embeddings, dim = dim)),
        label=train[constants.SUBJECTS],
    )

    val_pool = Pool(
        data=val["ABSTRACT"].apply(lambda text: text2vec(text, embeddings, dim = dim)),
        label=val[constants.SUBJECTS],
    )

    model = CatBoostClassifier(
        iterations = 100,
        loss_function='MultiLogloss',
        eval_metric='Accuracy',
        class_names=train_labels.columns.values
    )

    model.fit(train_pool, eval_set=val_pool, plot = True, use_best_model = True)
    return model

# Single-label Classification

def train_singlelabel(data):

    data = data.sample(frac=1)
    data = data.reset_index(drop=True)

    train, val = train_test_split(
        data,
        train_size=0.8,
        random_state=0,
        stratify=data['SUBJECT']
    )

    train_pool = Pool(
        data=train[["ABSTRACT"]],
        label=train["SUBJECT"],
        text_features=['ABSTRACT']
    )

    val_pool = Pool(
        data=val[["ABSTRACT"]],
        label=val["SUBJECT"],
        text_features=['ABSTRACT']
    )

    # Frequencies are used to counter class imbalances during model training

    value_counts = data["SUBJECT"].value_counts()
    frequencies = value_counts / value_counts.sum()

    model = CatBoostClassifier(
        iterations=100, 
        eval_metric='Accuracy', 
        class_names=frequencies.index.values,
        class_weights=frequencies, 
        tokenizers=TOKENIZERS
    )

    model.fit(train_pool, eval_set=val_pool, plot = True, use_best_model = True)
    return model

def prepare_tfidf(data):

    data = data.drop_duplicates(subset=["ABSTRACT"])
    data["PROCESSED"] = data["ABSTRACT"].apply(prepare_string)

    tfidf_vectorizer = TfidfVectorizer(min_df = 20, max_df = 0.7, max_features = 2048, ngram_range = (1,2))
    tfidf_vectorizer.fit(data["PROCESSED"])

    pickle.dump(tfidf_vectorizer, open(constants.CATBOOST_TFIDF, "wb"))

def train_singlelabel_tfidf(data): 

    data = data.sample(frac=1)
    data = data.reset_index(drop=True)

    data["PROCESSED"] = data["ABSTRACT"].apply(prepare_string)

    train, val = train_test_split(
        data,
        train_size=0.8,
        random_state=0,
        stratify=data['SUBJECT']
    )

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    vectorizer = pickle.load(open(constants.CATBOOST_TFIDF, 'rb'))

    X_train = vectorizer.transform(train["PROCESSED"])
    X_val = vectorizer.transform(val["PROCESSED"])

    train_pool = Pool(
        data=X_train,
        label=train["SUBJECT"],
    )

    val_pool = Pool(
        data=X_val,
        label=val["SUBJECT"],
    )

    model = CatBoostClassifier(iterations=100, eval_metric='Accuracy')
    model.fit(train_pool, eval_set=val_pool, plot = True, use_best_model = True)
    return model

def prepare_dictionary(): 

    text = files.load_txt(str(constants.PROCESSED_DIR / "all_abstracts_big.txt"))
    tokenized_text = prepare_tokens(text)

    dictionary = Dictionary(
        occurence_lower_bound=0,
        max_dictionary_size = 5000,
    )

    dictionary.fit(tokenized_text)
    dictionary.save(constants.CATBOOST_DICTIONARY)

def bag_of_words(tokenized_text, dictionary):

    nr_texts = len(tokenized_text)
    vocabulary_size = dictionary.size

    features = np.zeros((nr_texts, vocabulary_size))
    dict_indices = dictionary.apply(tokenized_text)

    for index in range(nr_texts):
        indices = dict_indices[index]
        features[index, indices] = 1

    return features

def train_singlelabel_dictionary(data): 

    data = data.sample(frac=1)
    data = data.reset_index(drop=True)

    dictionary = Dictionary()
    dictionary.load(constants.CATBOOST_DICTIONARY)

    train, val = train_test_split(
        data,
        train_size=0.8,
        random_state=0,
        stratify=data['SUBJECT']
    )

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    X_train = bag_of_words(train["ABSTRACT"].apply(prepare_tokens), dictionary)
    X_val = bag_of_words(val["ABSTRACT"].apply(prepare_tokens), dictionary)

    train_pool = Pool(
        data=X_train,
        label=train["SUBJECT"],
    )

    val_pool = Pool(
        data=X_val,
        label=val["SUBJECT"],
    )

    model = CatBoostClassifier(iterations=100, eval_metric='Accuracy')
    model.fit(train_pool, eval_set=val_pool, plot = True, use_best_model = True)
    return model

def train_singlelabel_embeddings(data):

    data = data.sample(frac=1)
    data = data.reset_index(drop=True)

    embeddings, dim = load_embeddings(constants.STARSPACE_EMBEDDINGS)
    data["EMBEDDING"] = data["ABSTRACT"].apply(lambda text: text2vec(text, embeddings, dim = dim))

    train, val = train_test_split(
        data,
        train_size=0.8,
        random_state=0
    )

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    train_pool = Pool(
        data=train["EMBEDDING"],
        label=train["SUBJECT"],
    )

    val_pool = Pool(
        data=val["EMBEDDING"],
        label=val["SUBJECT"],
    )

    value_counts = data["SUBJECT"].value_counts()
    frequencies = value_counts / value_counts.sum()

    model = CatBoostClassifier(
        iterations=100, 
        eval_metric='Accuracy', 
        class_names=frequencies.index.values,
        class_weights=frequencies
    )

    model.fit(train_pool, eval_set=val_pool, plot = True, use_best_model = True)
    return model