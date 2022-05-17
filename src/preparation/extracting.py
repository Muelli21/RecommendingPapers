import os
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

import utils.files as files
import utils.constants as constants
import utils.text_processing as text_processing

# Extracting Arxiv Papers

arxiv_categories = {
    "cs.AI": ("Computer Science", "Artificial Intelligence"), 
    "cs.AR": ("Computer Science", "Hardware Architecture"),
    "cs.CC": ("Computer Science", "Computational Complexity"),
    "cs.CE": ("Computer Science", "Computational Engineering"),
    "cs.CG": ("Computer Science", "Computational Geometry"),
    "cs.CL": ("Computer Science", "Computationa and Language"),
    "cs.CR": ("Computer Science", "Cryptography"),
    "cs.CV": ("Computer Science", "Computer Vision"),
    "cs.CY": ("Computer Science", "Computers and Society"),
    "cs.DB": ("Computer Science", "Databases"),
    "cs.DC": ("Computer Science", "Parallel Computing"),
    "cs.DL": ("Computer Science", "Digital Libraries"),
    "cs.DM": ("Computer Science", "Discrete Mathematics"),
    "cs.DS": ("Computer Science", "Data Structures and Algorithms"),
    "cs.ET": ("Computer Science", "Emerging Technologies"),
    "cs.FL": ("Computer Science", "Formal Languages"),
    "cs.GL": ("Computer Science", "General Literature"),
    "cs.GR": ("Computer Science", "Graphics"),
    "cs.GT": ("Computer Science", "Game Theory"),
    "cs.HC": ("Computer Science", "Human Computer Interaction"),
    "cs.IR": ("Computer Science", "Information Retrieval"),
    "cs.IT": ("Computer Science", "Information Theory"),
    "cs.LG": ("Computer Science", "Machine Learning"),
    "cs.LO": ("Computer Science", "Logic"),
    "cs.MA": ("Computer Science", "Multiagent Systems"),
    "cs.MM": ("Computer Science", "Multimedia"),
    "cs.MS": ("Computer Science", "Mathematical Software"),
    "cs.NA": ("Computer Science", "Numerical Analysis"),
    "cs.NE": ("Computer Science", "Neural and Evolutionary Computing"),
    "cs.NI": ("Computer Science", "Networking"),
    "cs.OH": ("Computer Science", "Other"),
    "cs.OS": ("Computer Science", "Operating Systems"),
    "cs.PF": ("Computer Science", "Performance"),
    "cs.PL": ("Computer Science", "Programming Languages"),
    "cs.RO": ("Computer Science", "Robotics"),
    "cs.SC": ("Computer Science", "Symbolic Computation"),
    "cs.SD": ("Computer Science", "Sound"),
    "cs.SE": ("Computer Science", "Software Engineering"),
    "cs.SI": ("Computer Science", "Social and Information Networks"),
    "cs.SY": ("Computer Science", "Systems and Control"),

    "econ.EM": ("Economics", "Econometrics"),
    "econ.GN": ("Economics", "General Economics"),
    "econ.TH": ("Economics", "Theoretical Economics"),

    "eess.AS": ("Electrical Engineering", "Audio and Speech Processing"),
    "eess.IV": ("Electrical Engineering", "Image and Video Processing"),
    "eess.SP": ("Electrical Engineering", "Signal Processing"),
    "eess.IV": ("Electrical Engineering", "Systems and Control"),

    "math.AC": ("Mathematics", "Commutative Algebra"),
    "math.AG": ("Mathematics", "Algebraic Geometry"),
    "math.AP": ("Mathematics", "Analysis of PDEs"),
    "math.AT": ("Mathematics", "Algebraic Topology"),
    "math.CA": ("Mathematics", "Classical Analysis"),
    "math.CO": ("Mathematics", "Combinatorics"),
    "math.CT": ("Mathematics", "Category Theory"),
    "math.CV": ("Mathematics", "Complex Variables"),
    "math.DG": ("Mathematics", "Differential Geometry"),
    "math.DS": ("Mathematics", "Dynamic Systems"),
    "math.FA": ("Mathematics", "Functional Analysis"),
    "math.GM": ("Mathematics", "General Mathematics"),
    "math.GN": ("Mathematics", "General Topology"),
    "math.GR": ("Mathematics", "Group Theory"),
    "math.GT": ("Mathematics", "Geometric Topoligy"),
    "math.HO": ("Mathematics", "History and Overview"),
    "math.IT": ("Mathematics", "Information Theory"),
    "math.KT": ("Mathematics", "K-Theory and Homology"),
    "math.LO": ("Mathematics", "Logic"),
    "math.MG": ("Mathematics", "Metric Geometry"),
    "math.MP": ("Mathematics", "Mathematical Physics"),
    "math.NA": ("Mathematics", "Numerical Analysis"),
    "math.NT": ("Mathematics", "Number Theory"),
    "math.OA": ("Mathematics", "Operator Algebras"),
    "math.OC": ("Mathematics", "Optimization and Control"),
    "math.PR": ("Mathematics", "Probability"),
    "math.QA": ("Mathematics", "Quantum Algebra"),
    "math.RA": ("Mathematics", "Rings and Algebras"),
    "math.RT": ("Mathematics", "Representation Theory"),
    "math.SG": ("Mathematics", "Symplectic Geometry"),
    "math.SP": ("Mathematics", "Spectral Theory"),
    "math.ST": ("Mathematics", "Statistics Theory"),

    "astro-ph.CO": ("Physics", "Astrophysics", "Cosmology and Nongalactic Astrophysics"),
    "astro-ph.EP": ("Physics", "Astrophysics", "Earth and Planetary Astrophysics"),
    "astro-ph.GA": ("Physics", "Astrophysics", "Astrophysics of Galaxies"),
    "astro-ph.HE": ("Physics", "Astrophysics", "High Energy Astrophysical Phenomena"),
    "astro-ph.IM": ("Physics", "Astrophysics", "Instrumentation and Methods"),
    "astro-ph.SR": ("Physics", "Astrophysics", "Solar and Stellar Astrophysics"),

    "cond-mat.dis-nn": ("Physics", "Condensed Matter", "Disordered Systems and Neural Networks"),
    "cond-mat.mes-hall": ("Physics", "Condensed Matter", "Mesoscale and Nanoscale Physics"),
    "cond-mat.mtrl-sci": ("Physics", "Condensed Matter", "Materials Science"),
    "cond-mat.other": ("Physics", "Condensed Matter", "Other Condensed Matter"),
    
    "cond-mat.quant-gas": ("Physics", "Condensed Matter", "Quantum Gases"),
    "cond-mat.soft": ("Physics", "Condensed Matter", "Soft Condensed Matter"),
    "cond-mat.stat-mech": ("Physics", "Condensed Matter", "Statistical Mechanics"),
    "cond-mat.str-el": ("Physics", "Condensed Matter", "Strongly Correlated Electrons"),
    "cond-mat.supr-con": ("Physics", "Condensed Matter", "Superconductivity"),

    "gr-qc": ("Physics", "General Relativity and Quantum Cosmology", "General Relativity and Quantum Cosmology"),

    "hep-ex": ("Physics", "High Energy Physics", "High Energy Physics - Experiment"),
    "hep-lat": ("Physics", "High Energy Physics", "High Energy Physics - Lattice"),
    "hep-ph": ("Physics", "High Energy Physics", "High Energy Physics - Phenomenology"),
    "hep-th": ("Physics", "High Energy Physics", "High Energy Physics - Theory"),

    "math-ph": ("Physics", "Mathematical Physics", "Mathematical Physics"),

    "nlin.AO": ("Physics", "Nonlinear Sciences", "Adaptation and Self-Organizing Systems"),
    "nlin.CD": ("Physics", "Nonlinear Sciences", "Chaotic Dynamics"),
    "nlin.CG": ("Physics", "Nonlinear Sciences", "Cellular Automata and Lattice Gases"),
    "nlin.PS": ("Physics", "Nonlinear Sciences", "Pattern Formation and Solitons"),
    "nlin.SI": ("Physics", "Nonlinear Sciences", "Exactly Solvable and Integrable Systems"),

    "nucl-ex": ("Physics", "Nuclear", "Nuclear Experiment"),
    "nucl-th": ("Physics", "Nuclear", "Nuclear Theory"),

    "physics.acc-ph": ("Physics", "Physics", "Accelerator Physics"),
    "physics.ao-ph": ("Physics", "Physics", "Atmospheric and Oceanic Physics"),
    "physics.app-ph": ("Physics", "Physics", "Applied Physics"),
    "physics.atm-clus": ("Physics", "Physics", "Atomic and Molecular Clusters"),
    "physics.atom-ph": ("Physics", "Physics", "Atomic Physics"),
    "physics.bio-ph": ("Physics", "Physics", "Biological Physics"),
    "physics.chem-ph": ("Physics", "Physics", "Chemical Physics"),
    "physics.class-ph": ("Physics", "Physics", "Classical Physics"),
    "physics.comp-ph": ("Physics", "Physics", "Computational Physics"),
    "physics.data-an": ("Physics", "Physics", "Data Analysis, Statistics and Probability"),
    "physics.ed-ph": ("Physics", "Physics", "Physics Education"),
    "physics.flu-dyn": ("Physics", "Physics", "Fluid Dynamics"),
    "physics.gen-ph": ("Physics", "Physics", "General Physics"),
    "physics.geo-ph": ("Physics", "Physics", "Geophysics"),
    "physics.hist-ph": ("Physics", "Physics", "Hisotry and Philosophy of Physics"),
    "physics.ins-det": ("Physics", "Physics", "Instrumentation and Detectors"),
    "physics.med-ph": ("Physics", "Physics", "Medical Physics"),
    "physics.optics": ("Physics", "Physics", "Optics"),
    "physics.plasm-ph": ("Physics", "Physics", "Plasma Physics"),
    "physics.pop-ph": ("Physics", "Physics", "Popular Physics"),
    "physics.soc-ph": ("Physics", "Physics", "Physics and Society"),
    "physics.space-ph": ("Physics", "Physics", "Space Physics"),
    
    "quant-ph": ("Physics", "Quantum Physics", "Quantum Physics"),

# Actually, this is domain is called Quantitative Biology, but we merge it with Biology
    "q-bio.BM": ("Quantitative Biology", "Biomolecules"),
    "q-bio.CB": ("Quantitative Biology", "Behavior"),
    "q-bio.GN": ("Quantitative Biology", "Genomics"),
    "q-bio.MN": ("Quantitative Biology", "Molecular Networks"),
    "q-bio.NC": ("Quantitative Biology", "Neurons and Cognition"),
    "q-bio.OT": ("Quantitative Biology", "Other Quantitative Biology"),
    "q-bio.PE": ("Quantitative Biology", "Populations and Evolution"),
    "q-bio.QM": ("Quantitative Biology", "Quantitative Methods"),
    "q-bio.SC": ("Quantitative Biology", "Subcellular Processes"),
    "q-bio.TO": ("Quantitative Biology", "Tissues and Organs"),

# NOTE: It might make sense to merge this with Economics or Business
    "q-fin.CP": ("Quantitative Finance", "Computational Finance"),
    "q-fin.EC": ("Quantitative Finance", "Economics"),
    "q-fin.GN": ("Quantitative Finance", "General Finance"),
    "q-fin.MF": ("Quantitative Finance", "Mathematical Finance"),
    "q-fin.PM": ("Quantitative Finance", "Portfolio Management"),
    "q-fin.PR": ("Quantitative Finance", "Pricing of Securities"),
    "q-fin.RM": ("Quantitative Finance", "Risk Management"),
    "q-fin.ST": ("Quantitative Finance", "Statistial Finance"),
    "q-fin.TR": ("Quantitative Finance", "Trading and Market Microstructure"),

    "stat.AP": ("Statistics", "Applications"),
    "stat.CO": ("Statistics", "Computation"),
    "stat.ME": ("Statistics", "Methodology"),
    "stat.ML": ("Statistics", "Machine Learning"),
    "stat.OT": ("Statistics", "Other Statistics"),
    "stat.TH": ("Statistics", "Statistics Theory"),
}

def extract_arxiv_categories(text):
    """
    Extracts and translates arxiv categories to subjects

    Params: 
        text: arxiv category as string

    Returns: 
        tuple representing arxiv subject
    """

    text = text.replace("[", "").replace("]", "").replace("'","")
    terms = text.split(", ")
    return [arxiv_categories[term] for term in terms if term in arxiv_categories]

def process_arxiv_abstracts():
    """
    Processes arxiv dataset files and encodes subjects using one hot encoding

    Returns: 
        dataframe(columns = ["TITLE", "ABSTRACT", SUBJECTS...])
    """

    data1 = pd.read_csv(constants.DATA_ARXIV)
    data1 = data1.rename(columns={"titles": "TITLE", "summaries": "ABSTRACT", "terms": "SUBJECTS"})

    data2 = pd.read_csv(constants.DATA_ARXIV2)
    data2 = data2.rename(columns={"titles": "TITLE", "abstracts": "ABSTRACT", "terms": "SUBJECTS"})

    data = pd.concat([data1, data2], ignore_index=True)
    data["SUBJECTS"] = data["SUBJECTS"].apply(extract_arxiv_categories)
    data["ABSTRACT"] = data["ABSTRACT"].apply(text_processing.remove_whitespace)

    data = data.drop_duplicates(subset=["TITLE"])
    data = data.reset_index(drop=True)

    # Encodes subjects using one hot encoding

    rows = data.shape[0]
    columns = len(constants.SUBJECTS)
    entries = np.zeros((rows, columns), dtype=int)

    for index, values in data["SUBJECTS"].iteritems():
        for subject_wrapper in values: 
            subject = subject_wrapper[0]
            subject_index = constants.SUBJECTS2INDEX[subject]
            entries[index, subject_index] = 1

    subjects_frame = pd.DataFrame(entries, columns = constants.SUBJECTS)
    export_frame = pd.concat([data.iloc[:, :2], subjects_frame], axis = 1)
    return export_frame

def process_arxiv_papers(): 
    """
    Processes arxiv dataset files and encodes subjects using one hot encoding

    Returns: 
        dataframe(columsn = ["TITLE", "ABSTRACT", SUBJECTS...])
    """

    papers = []

    for paper in files.load_jsonl(constants.DATA_ARXIV_FULL):
        title = paper["title"]
        abstract = text_processing.remove_whitespace(paper["abstract"])
        subjects = extract_arxiv_categories(paper["categories"])

        papers.append((title, abstract, subjects))

    data = pd.DataFrame(papers, columns = ["TITLE", "ABSTRACT", "SUBJECTS"])
    data = data.drop_duplicates(subset=["ABSTRACT"])
    data = data.reset_index(drop=True)

    rows = data.shape[0]
    columns = len(constants.SUBJECTS)
    entries = np.zeros((rows, columns), dtype=int)

    for index, values in data["SUBJECTS"].iteritems():
        for subject_wrapper in values: 
            subject = subject_wrapper[0]
            subject_index = constants.SUBJECTS2INDEX[subject]
            entries[index, subject_index] = 1

    subjects_frame = pd.DataFrame(entries, columns = constants.SUBJECTS)
    export_frame = pd.concat([data.iloc[:, :2], subjects_frame], axis = 1)
    return export_frame

# Extracting Kaggle papers

kaggle_labels = {
    "cs": "Computer Science",
    "stat": "Statistics",
    "physics": "Physics",
    "math": "Mathematics",
}

def process_kaggle():
    """
    Processes kaggle dataset files and encodes subjects using one hot encoding

    Returns: 
        dataframe(columns = ["TITLE", "ABSTRACT", SUBJECTS...])
    """

    # Topic Modelling 1

    data1 = pd.read_csv(constants.DATA_KAGGLE1)

    categories1 = data1.iloc[:, 3:]
    data1 = data1.drop(data1.columns[3:], axis=1)

    categories = categories1.idxmax(axis=1)

    data1["SUBJECT"] = categories
    data1 = data1.drop(["ID"], axis = 1)

    # Topic Modelling 2

    data2 = pd.read_csv(constants.DATA_KAGGLE2)

    categories2 = data2.iloc[:, 2:]
    data2 = data2.drop(data2.columns[2:], axis=1)

    categories2 = categories2.idxmax(axis=1)

    data2["SUBJECT"] = categories2
    data2 = data2.drop(["id"], axis = 1)
    data2["TITLE"] = None
    data2 = data2[["TITLE", "ABSTRACT", "SUBJECT"]]

    # Topic Modelling 3

    data3 = pd.read_csv(constants.DATA_KAGGLE3_FEATURES)
    categories3 = pd.read_csv(constants.DATA_KAGGLE3_LABELS)

    data3 = data3.merge(categories3, left_on="id", right_on="id")
    data3 = data3.drop(["id"], axis = 1)
    data3["TITLE"] = None
    data3 = data3.rename(columns={"abstract": "ABSTRACT", "category": "SUBJECT"})
    data3 = data3[["TITLE", "ABSTRACT", "SUBJECT"]]
    data3["SUBJECT"] = data3["SUBJECT"].apply(lambda label: kaggle_labels[label])

    # Merging data 

    data = pd.concat([data1, data2, data3], ignore_index=True)
    data = data[data["ABSTRACT"].str.contains("withdrawn by the author")==False]
    data = data[data["ABSTRACT"].str.contains("This paper has been withdrawn.")==False]
    data = data.reset_index(drop=True)

    data["ABSTRACT"] = data["ABSTRACT"].apply(text_processing.remove_whitespace)

    subjects = data["SUBJECT"].apply(lambda label: [label])

    mlb = MultiLabelBinarizer()
    result = pd.DataFrame(mlb.fit_transform(subjects), columns=mlb.classes_, index=subjects.index)

    onehot = data.drop("SUBJECT", axis = 1)
    onehot = pd.concat([onehot, result], axis = 1)
    return onehot

# Extracting Elsevier Papers

elsevier_labels = {
    "AGRI": "Agricultural Science", 
    "ARTS": "Arts and Humanities", 
    "BIOC": "Biochemistry",
    "BUSI": "Business", 
    "CENG": "Chemical Engineering", 
    "CHEM": "Chemistry", 
    "COMP": "Computer Science", 
    "DECI": "Decision Sciences", 
    "DENT": "Dentistry", 
    "EART": "Earth and Planetary Sciences", 
    "ECON": "Economics", 
    "ENER": "Energy Sciences", 
    "ENGI": "Engineering", 
    "ENVI": "Environmental Science", 
    "HEAL": "Health Professions", 
    "IMMU": "Immunology", 
    "MATE": "Material Sciences", 
    "MATH": "Mathematics", 
    "MEDI": "Medical Science", 
    "MULT": "MULT", 
    "NEUR": "Neuroscience", 
    "NURS": "Nursing", 
    "PHAR": "Pharmacy", 
    "PHYS": "Physics", 
    "PSYC": "Psychology", 
    "SOCI": "Social Sciences", 
    "VETE": "Veterinary Science"
}

def clean_elsevier_subjects(subjects):
    """
    Extracts and translates elsevier categories to subjects

    Params: 
        subjects: list of elsevier categories

    Returns: 
        list of subjects
    """

    return [elsevier_labels[subject] for subject in subjects ]

def process_elsevier(): 
    """
    Processes elsevier dataset files and encodes subjects using one hot encoding

    Returns: 
        dataframe(columns = ["TITLE", "ABSTRACT", SUBJECTS...])
    """

    directory = constants.DATA_ELSEVIER_DIR
    papers = []

    for filename in os.listdir(directory):

        paper = files.load_json(str(directory / filename))
        metadata = paper["metadata"]

        if "title" in metadata and "abstract" in paper and "subjareas" in metadata:
            
            information = {
                "TITLE": metadata["title"],
                "ABSTRACT": paper["abstract"],
                "SUBJECTS" : metadata["subjareas"],
                "KEYWORDS" : metadata["keywords"] if "keywords" in metadata else None,
            }

            papers.append(information)

    data = pd.DataFrame(papers)
    data["ABSTRACT"] = data["ABSTRACT"].apply(text_processing.remove_whitespace)
    data["SUBJECTS"] = data["SUBJECTS"].apply(clean_elsevier_subjects)

    subjects = data["SUBJECTS"]
    mlb = MultiLabelBinarizer()
    result = pd.DataFrame(mlb.fit_transform(subjects), columns=mlb.classes_, index=subjects.index)

    onehot = data.drop("SUBJECTS", axis = 1)
    onehot = pd.concat([onehot, result], axis = 1)
    return onehot

# Extracting Web of Science

web_labels = {
    "CS": "Computer Science",
    "Civil": "Civil Engineering",
    "ECE": "Electrical Engineering",
    "biochemistry": "Biochemistry",
    "Medical": "Medical Science",
    "MAE": "Engineering",
    "Psychology": "Psychology"
}

def clean_web_keywords(text):
    """
    Cleans web of science keywords

    Returns: 
        list of keywords
    """

    text = text.strip()
    keywords = text.split(";")
    return keywords

def process_web():
    """
    Processes web of science dataset files and encodes subjects using one hot encoding

    Returns: 
        dataframe(columns = ["TITLE", "ABSTRACT", SUBJECTS...])
    """

    # Topic Modelling Web of Science

    data = pd.read_excel(constants.DATA_WOS)
    data["SUBJECT"] = data["Domain"].apply(lambda label: web_labels[label.strip()])

    data = data.drop(["Y1", "Y2", "Y", "Domain"], axis = 1)
    data = data.rename(columns={"area": "AREA", "Abstract": "ABSTRACT", "keywords": "KEYWORDS"})
    
    data["TITLE"] = None
    data["KEYWORDS"] = data["KEYWORDS"].apply(clean_web_keywords)
    data["ABSTRACT"] = data["ABSTRACT"].apply(text_processing.remove_whitespace)

    data = data[["TITLE", "ABSTRACT", "SUBJECT", "AREA", "KEYWORDS"]]

    subjects = data["SUBJECT"].apply(lambda label: [label])

    mlb = MultiLabelBinarizer()
    result = pd.DataFrame(mlb.fit_transform(subjects), columns=mlb.classes_, index=subjects.index)

    onehot = data.drop("SUBJECT", axis = 1)
    onehot = pd.concat([onehot, result], axis = 1)
    return onehot

# Extracting S2ORC Papers

s2orc_labels = {
    "Art": "Arts and Humanities",
    "Materials Science": "Material Sciences",
    "Medicine": "Medical Science",
    "Sociology": "Social Sciences"
}

def clean_s2orc_categories(subjects):
    """
    Extracts and translates s2orc categories to subjects

    Params: 
        subjects: list of s2orc categories

    Returns: 
        list of subjects
    """

    if subjects is None: 
        return None

    return [s2orc_labels[subject] if subject in s2orc_labels else subject for subject in subjects ]

def process_s2orc(file_format = ".csv", overwrite = False):

    source_path = constants.SOURCE_PATH_S2ORC
    processed_path = constants.PROCESSED_S2ORC_DIR

    for filename in os.listdir(source_path):
        if ".gz" in filename:
            csv_name = filename.split(".")[0] + file_format

            if not overwrite and os.path.isfile(processed_path / csv_name):
                print("Skipping already existing " + filename)
                continue

            print("Processing " + filename)

            papers = []

            for paper in files.load_compressed_jsonl(str(source_path / filename)):
                abstract = paper["abstract"]
                subjects = paper["mag_field_of_study"]

                if abstract is not None and subjects is not None:
                    title = paper["title"]
                    paper_id = paper["paper_id"]
                    papers.append((paper_id, title, abstract, subjects))

            data = pd.DataFrame(papers, columns = ["PAPER_ID", "TITLE", "ABSTRACT", "SUBJECTS"])
            data["SUBJECTS"] = data["SUBJECTS"].apply(clean_s2orc_categories)
            data.drop_duplicates(subset=["ABSTRACT"])

            if file_format == ".csv": 
                data.to_csv(str(processed_path / csv_name), index = False)

            if file_format == ".parquet": 
                data.to_parquet(str(processed_path / csv_name), index = False)

            del papers
            del data

            gc.collect()

# General

def extract_column_as_txt(data, file_path, column = "ABSTRACT"):
    """
    Stores all abstracts of a data frame in a .txt-file.
    Each abstract is stored in its own line

    Params: 
        data: data frame with a column named "ABSTRACT"
        file_path: file path to txt-file
        column: name of the column to be extracted
    """

    with open(file_path, 'a+') as file:
        for abstract in data[column]:
            abstract = text_processing.remove_whitespace(abstract)
            file.write(abstract + '\n')

def merge(include_origin = True):
    """
    Merges kaggle, web of science, arxiv and elsevier data into one dataframe

    Params: 
        include_origin: boolean indicating whether a column storing each rows origin should be included

    Returns: 
        dataframe(columns = ["TITLE", "ABSTRACT", SUBJECTS...])
    """

    data_kaggle = process_kaggle()
    data_web = process_web()
    data_arxiv = process_arxiv_abstracts()
    data_arxiv2 = process_arxiv_papers()
    data_elsevier = process_elsevier()

    if include_origin: 
        data_kaggle["ORIGIN"] = "kaggle"
        data_web["ORIGIN"] = "webOfScience"
        data_arxiv["ORIGIN"] = "arxiv"
        data_arxiv2["ORIGIN"] = "arxiv"
        data_elsevier["ORIGIN"] = "elsevier"

    data = pd.concat([data_kaggle, data_arxiv, data_arxiv2, data_web, data_elsevier], ignore_index=True)
    data[constants.SUBJECTS] = data[constants.SUBJECTS].fillna(value = 0)
    data[constants.SUBJECTS] = data[constants.SUBJECTS].astype(int)

    data = data.sample(frac=1)
    data = data.drop_duplicates(subset=["ABSTRACT"], keep = "last")
    data = data[["TITLE", "ABSTRACT", "ORIGIN"] + constants.SUBJECTS]
    data = data.reset_index(drop=True)
    return data