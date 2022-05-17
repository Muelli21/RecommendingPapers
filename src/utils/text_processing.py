import csv
import regex as re
import numpy as np
import pandas as pd
from gensim import models
from catboost.text_processing import Tokenizer
import utils.constants as constants

REGEX_WORD_BREAK = re.compile(r"-\n")
REGEX_NEWLINE = re.compile(r"(?:\r\n|\r|\n)")
REGEX_WHITESPACE = re.compile(r"\s\s+")
REGEX_SPECIAL_CHARS = re.compile(r"[0123456789?€!@#$—ツ►๑۩۞۩•*”˜˜”*°°*``,.;:(){}]")

REGEX_ISBN = re.compile(r"^(?:ISBN(?:-10)?:? )?(?=[0-9X]{10}$|(?=(?:[0-9]+[- ]){3})[- 0-9X]{13}$)[0-9]{1,5}[- ]?[0-9]+[- ]?[0-9]+[- ]?[0-9X]$")
REGEX_EMAIL = re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
REGEX_URL = re.compile(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()!@:%_\+.~#?&\/\/=]*)")

tokenizer = Tokenizer(
    lowercasing=True,
    separator_type='BySense',
    token_types=['Word', 'Number']
)

def clean_string(text, remove_special_chars = False):
    """
    Cleans the given text by removing line breaks within words, line breaks, emails, and urls

    Params:
        text: text string to be cleaned

    Returns:
        string: cleaned text
    """

    text = REGEX_WORD_BREAK.sub("", text)
    text = REGEX_NEWLINE.sub("", text)

    text = REGEX_URL.sub(" ", text)
    text = REGEX_EMAIL.sub(" ", text)
    text = REGEX_ISBN.sub(" ", text)

    text = REGEX_WHITESPACE.sub(" ", text)

    if remove_special_chars: 
        text = REGEX_SPECIAL_CHARS.sub(" ", text)

    return text

def remove_whitespace(string):
    """
    Removes linebreaks and excess whitespace

    Params: 
        string: string to be cleaned

    Returns: 
        cleaned string
    """

    string = REGEX_NEWLINE.sub(" ", string)
    string = REGEX_WHITESPACE.sub(" ", string)
    return string

def tokenize(string):
    """
    Tokenizes a string using CatBoost's bySense tokenizer and removes stopwords

    Params: 
        string: string to be tokenized

    Returns: 
        list of tokens
    """

    return [ token for token in tokenizer.tokenize(string) if token not in constants.STOP_WORDS ]

def prepare_tokens(string):
    """
    Prepares a string by cleaning and tokenizing it

    Params: 
        string: string to be cleaned and tokenized

    Returns: 
        list of tokens
    """

    string = clean_string(string)
    tokens = tokenize(string)
    return tokens

def prepare_string(string):
    """
    Prepares a string by cleaning and tokenizing it

    Params: 
        string: string to be cleaned and tokenized

    Returns: 
        cleaned string
    """

    string = clean_string(string)
    tokens = tokenize(string)
    return " ".join(tokens)

def load_embeddings(embeddings_path):
    """
    Loads pre-trained word embeddings
    NOTE: Depends on the file extension to choose the appropriate embeddings wrapper
    
    Params:
        embeddings_path: path to the embeddings file
    
    Returns:
        embeddings: dict mapping words to vectors
        embeddings_dim: dimension of the vectors
    """

    if ".bin" in embeddings_path: 
        return load_gensim_embeddings(embeddings_path)

    if ".tsv" in embeddings_path: 
        return load_tsv_embeddings(embeddings_path)

def load_tsv_embeddings(embeddings_path):
    """
    Loads pre-trained word embeddings from tsv file
    
    Params:
        embeddings_path: path to the embeddings file
    
    Returns:
        embeddings: dict mapping words to vectors
        embeddings_dim: dimension of the vectors
    """

    df = pd.read_csv(embeddings_path, delimiter = "\t", header = None, quoting=csv.QUOTE_NONE)
    words = df.iloc[:, 0].values
    vectors = df.iloc[:, 1:].values
    embeddings = dict(zip(words, vectors))
    return embeddings, vectors.shape[1]

def load_gensim_embeddings(embeddings_path):
    """
    Loads word2vec representations from a binary file. E.g. the following embeddings:
    [Google embeddings](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g) 

    Params:
        embeddings_path: path to the embeddings file
    
    Returns:
        embeddings: dict mapping words to vectors
        embeddings_dim: dimension of the vectors
    """
    embeddings = models.KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
    return embeddings, embeddings.vector_size

def text2vec(text, embeddings, dim):
    """
    Splits a given text into tokens (= words) and calculates the mean of word vectors to represent the text as a vector.

    Params: 
        text: string to vectorize
        embeddings: mapping of words to vectors

    Returns: 
        mean word embedding over the input string
    """

    if text is None: 
        return  np.zeros(dim)

    tokens = text.split(" ")
    word_vectors = [ embeddings[token] for token in tokens if token in embeddings and token not in constants.STOP_WORDS]
    document_average = np.mean(word_vectors, axis = 0) if word_vectors else np.zeros(dim)
    return document_average

def text2vec_adjusted(text, embeddings, dim, subject, subject_embeddings):
    """
    Splits a given text into tokens (= words) and calculates the mean of word vectors to represent the text as a vector.
    To adjust for subject specific vocabulary, the text's subject embedding as the average text embedding over a subject is 
    subtracted from the usual text embedding.

    Params:
        text: string to vectorize
        embeddings: mapping of words to vectors
        dim: dimension of the word embeddings
        subject: text's subject
        subject_embeddings: mapping of subjects to vectors

    Returns: 
        mean word embedding over the input string minus the average subject embedding
    """

    # NOTE: Attention, there has to be at least one subject. Otherwise, using the 0th index won't work
    return text2vec(text, embeddings, dim) - subject_embeddings[subject]

def text_collection2vec(texts, embeddings, dim):
    """
    Computes the average paper embedding of a collection of texts.

    Params: 
        texts: texts of a certain subject
        wv_embeddings: word embeddings for vectorization
        dim: dimension of the word embeddings

    Returns: 
        mean word embedding over the input string
    """

    vectors = [ text2vec(text, embeddings, dim) for text in texts ]
    average = np.mean(vectors, axis = 0) if vectors else np.zeros(dim)
    return average
