import os
from pathlib import Path
from nltk.corpus import stopwords


# Directories
ROOT_DIR = Path(os.path.abspath(__file__ + "/../../../") + "/")
DATA_DIR = ROOT_DIR /  "data"
PDFS_DIR = ROOT_DIR / "data/pdfs"
DEPENDENCIES_DIR = ROOT_DIR / "dependencies"
DATASETS_DIR = ROOT_DIR / "data/datasets"
PROCESSED_DIR = ROOT_DIR / "data/processed"

# Embeddings
GOOGLE_EMBEDDINGS = str(ROOT_DIR / "data/embeddings/GoogleNews-vectors-negative300.bin")
STARSPACE_EMBEDDINGS = str(ROOT_DIR / "data/embeddings/starspace_embedding_200.tsv")
SUBJECT_EMBEDDINGS = str(ROOT_DIR / "data/embeddings/subject_embeddings.tsv")

# CatBoost 
CATBOOST_DIR = ROOT_DIR / "data/models/CatBoost/"
SUBJECT_LABELING_MODEL_SINGLE = str(CATBOOST_DIR / "singlelabel.cbm")
SUBJECT_LABELING_MODEL_MULTI = str(CATBOOST_DIR / "multilabel.cbm")
CATBOOST_TFIDF = str(CATBOOST_DIR / "tfidf.pickle")
CATBOOST_DICTIONARY = str(CATBOOST_DIR / "dictionary.pickle")

# Stop words
STOP_WORDS = stopwords.words('english')

# Subject Files
BATCH_SIZE = 100_000
SOURCE_PATH_S2ORC = DATASETS_DIR / "S2ORC/metadata"
SOURCE_PATH_MERGED = str(PROCESSED_DIR / "merged.csv")
SUBJECT_FILES_DIR = PROCESSED_DIR / "subjects"
EMBEDDED_SUBJECT_FILES_DIR = PROCESSED_DIR / "embedded_subjects"
SUBJECT_INFO = str(PROCESSED_DIR / "subjects/info.csv")
PROCESSED_S2ORC_DIR = PROCESSED_DIR / "S2ORC"

# Upload Files
USER_UPLOAD_DIR = DATA_DIR / "uploads"
USER_UPLOAD = str(USER_UPLOAD_DIR / "user_input.pdf")
USER_UPLOAD_CERM = str(USER_UPLOAD_DIR / "user_input.cermxml")

# Data Sets
DATA_ARXIV = str(DATASETS_DIR / "Topic Modelling Arxiv/arxiv_data.csv")
DATA_ARXIV2 = str(DATASETS_DIR / "Topic Modelling Arxiv/arxiv_data2.csv")
DATA_ARXIV_FULL = str(DATASETS_DIR / "ARXIV/arxiv-metadata-oai-snapshot.json")

DATA_KAGGLE1 = str(DATASETS_DIR / "Topic Modelling 1/train.csv")
DATA_KAGGLE2 = str(DATASETS_DIR / "Topic Modelling 2/train.csv")
DATA_KAGGLE3_FEATURES = str(DATASETS_DIR / "Topic Modelling 3/data_input.csv")
DATA_KAGGLE3_LABELS = str(DATASETS_DIR / "Topic Modelling 3/data_output.csv")

DATA_ELSEVIER_DIR = DATASETS_DIR / "Topic Modelling Elsevier/json/json"

DATA_WOS = str(DATASETS_DIR / "Topic Modelling Web of Science/WebOfScience/Meta-data/Data.xlsx")

# Legacy
SUBJECT_HASHES = str(ROOT_DIR / "data/subject_hashes.json")
PAPERS = str(ROOT_DIR / "data/papers.json")
TREES = str(ROOT_DIR / "data/tree.json")

SUBJECTS = [
    "Agricultural Science", 
    "Arts and Humanities", 
    "Biochemistry",
    "Biology",
    "Business",
    "Civil Engineering",
    "Chemical Engineering", 
    "Chemistry", 
    "Computer Science", 
    "Decision Sciences", 
    "Dentistry", 
    "Earth and Planetary Sciences", 
    "Economics", 
    "Electrical Engineering", 
    "Energy Sciences", 
    "Engineering", 
    "Environmental Science",
    "Geology",
    "Geography", 
    "Health Professions",
    "History",
    "Immunology", 
    "Material Sciences",
    "Mathematics", 
    "Medical Science", 
    "MULT", 
    "Neuroscience", 
    "Nursing",
    "Political Science",
    "Pharmacy",
    "Philosophy",
    "Physics", 
    "Psychology", 
    "Quantitative Biology", 
    "Quantitative Finance", 
    "Social Sciences",
    "Statistics", 
    "Veterinary Science", 
]

SUBJECTS2INDEX = dict(zip(SUBJECTS, range(len(SUBJECTS))))

SUBJECTS2COLOR = {
    # Colors generated with https://mokole.com/palette.html
    "Agricultural Science": "#a0522d", #sienna
    "Arts and Humanities": "#ff00ff", #fuchsia
    "Biochemistry": "#98fb98", #palegreen
    "Biology" : "#808000", #olive
    "Business": "#1e90ff", #dodgerblue
    "Civil Engineering": "#ffffe0", #lightyellow
    "Chemical Engineering": "#7fff00", #chartreuse
    "Chemistry" : "#9acd32", #yellowgreen
    "Computer Science": "#dc143c", #crimson
    "Decision Sciences": "#00bfff", #deepskyblue 
    "Dentistry": "#778899", #lightslategray
    "Earth and Planetary Sciences": "#d2b48c", #tan
    "Economics": "#0000ff", #blue
    "Electrical Engineering": "#ffff00", #yellow
    "Energy Sciences": "#f0e68c", #khaki
    "Engineering": "#8b0000", #darkred
    "Environmental Science": "#dda0dd", #plum
    "Geology": "#b03060", #maroon3
    "Geography": "#da70d6", #orchid
    "Health Professions": "#00ff7f", #springgreen
    "History": "#a020f0", #purple3
    "Immunology": "#008b8b", #darkcyan
    "Material Sciences": "#daa520", #goldenrod
    "Mathematics": "#f08080", #lightcoral
    "Medical Science": "#3cb371", #mediumseagreen
    "MULT": "#add8e6", #lightblue
    "Neuroscience": "#2f4f4f", #darkslategray
    "Nursing": "#008000", #green
    "Political Science": "#6a5acd", #slateblue
    "Pharmacy": "#40e0d0", #turquoise
    "Philosophy": "#ff1493", #deeppink
    "Physics": "#ff8c00", #darkorange
    "Psychology": "#ffc0cb", #pink
    "Quantitative Biology": "#ff0000", #red
    "Quantitative Finance": "#000080", #navy
    "Social Sciences": "#800080", #purple
    "Statistics": "#ff7f50", #coral
    "Veterinary Science": "#556b2f", #darkolivegreen
    "None": "#ffffff"
}