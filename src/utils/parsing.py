import os

import hashlib
import io
from lxml import etree
from utils import constants

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

def extract_title(file_path:str):
    """
    Extracts a .pdf-file's title based on its XML representation.

    Params:
        file_path: The file path of the .pdf's XML representation

    Returns:
        String representation of the file's title if a title exists, None otherwise
    """

    doc = etree.parse(file_path)
    title = doc.find(".//title-group")
    return title.find("article-title").text if title is not None else None

def extract_abstract(file_path:str):
    """
    Extracts a .pdf-file's abstract based on its XML representation.

    Params:
        file_path: The file path of the .pdf's XML representation

    Returns:
        String representation of the file's abstract if an abstract exists, None otherwise
    """
    doc = etree.parse(file_path)
    abstract = doc.find(".//abstract")
    return abstract.find("p").text if abstract is not None else None

def extract_references(file_path:str):
    """
    Extracts a .pdf-file's references based on its XML representation.

    Params: 
        file_path: The file path of the .pdf's XML representation

    Returns:
        List of dictionaries with references
    """

    doc = etree.parse(file_path)
    references = doc.findall(".//ref")

    return [ parse_reference(reference) for reference in references ]

def parse_reference(ref_element):
    """
    Parses a reference from etree XML-Element-format to a dictionary representation 
    
    Params:
        ref_element: etree XML-Element of a reference

    Returns:
        Dictionary representation of a reference with fields 'title', 'authors', 'published', 'published-in'
    """
    parsed_ref = {}
    ref_citation = ref_element.find("mixed-citation")
    
    if ref_citation is not None:
        all_authors = []
        all_author_elements = ref_citation.findall("string-name")

        for author_element in all_author_elements:
            first_name = author_element.find("given-names").text if author_element.find("given-names") is not None else None
            surname = author_element.find("surname").text if author_element.find("surname") is not None else None

            if first_name is not None and surname is not None:
                full_name = first_name + " " + surname
                all_authors.append(full_name)

        parsed_ref['title'] = ref_citation.find("article-title").text if ref_citation.find("article-title") is not None else None
        parsed_ref['authors'] = all_authors if all_authors is not None else None
        parsed_ref['published'] = ref_citation.find("year").text if ref_citation.find("year") is not None else None
        parsed_ref['source'] = ref_citation.find("source").text if ref_citation.find("source") is not None else None

    return parsed_ref

def parse_pdfs_to_xml(pdfs_dir = constants.PDFS_DIR):
    """
    Parses all pdfs into the `/data/pdfs/`-directory using `cermine.jar`. 
    This produces `.cermxml`-files that contain the parsed XML-representations.

    Source: Cermine
        - Github: https://github.com/CeON/CERMINE
        - Jar: https://maven.ceon.pl/artifactory/repo/pl/edu/icm/cermine/cermine-impl/1.13/
    """
    
    working_directory = os.getcwd()
    os.chdir(constants.DEPENDENCIES_DIR)
    os.system("java -cp cermine.jar pl.edu.icm.cermine.ContentExtractor -path \"" + str(pdfs_dir) + "\"")
    os.chdir(pdfs_dir)
    os.system("rm -rf ./*/")
    os.chdir(working_directory)

def integrate_xml(papers, pdfs_dir = constants.PDFS_DIR):
    """
    Integrates the information extracted by cermine into the 
    dict that stores processed papers. 

    Params:
        directory_path: path of directory where XML-representations of .pdfs should be stored
        papers: dictionary containing all papers with hash as key and dictionary of attributes as value
    """
    for file_name in os.listdir(pdfs_dir):
        if not file_name.endswith('.cermxml'): continue
        
        path = str(pdfs_dir / file_name)
        Hash = file_name.split(".")[0]

        title = extract_title(path)
        abstract = extract_abstract(path)
        references = extract_references(path)

        papers[Hash] = {"title": title}
        papers[Hash]["abstract"]=abstract
        papers[Hash]["references"] = references

def parse_pdf(content, pdf_link):
    """
    Parses response.content and return pdf-contents as string. Returns dict of pdf properties.
    """
    # NOTE: pdfminer.six is the only pdf parser that allows to parse byte-streams instead of files. 
    # THis allows us to process the pdfs' contents without storing them
    
    # Hashing to have "unique" identifierts for the scraped files
    # The hashes can then be used e.g. to check for duplicate pdfs
    hasher = hashlib.sha1()
    hasher.update(content)

    output_string = io.StringIO()

    try: 
        with io.BytesIO(content) as in_file:
        
            parser = PDFParser(in_file)
            doc = PDFDocument(parser)

            rsrcmgr = PDFResourceManager()
            laparams = LAParams(all_texts = True) # all_texts = True to ensure proper spacing
            device = TextConverter(rsrcmgr, output_string, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            
            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)

            properties = {
                "url": pdf_link,
                "hash": hasher.hexdigest(),
                "title": None,
                "author": None,
                "creator": None, 
                "producer": None,
                "creationDate": None,
                "modDate": None,
                "keywords": None,
                "references": None,
                "referencesHashs": [],
                "referencedByHashs": [],
                "abstract": None,
                "content": output_string.getvalue(),
            }

            # Update values in properties dict
            for key, value in doc.info[0].items():

                # For all keys of interest, we decode the values and update the dict with them
                if key in properties.keys():
                    properties[key] = value.decode(errors="ignore")
                    
            return properties
    except: 
        print("Error: Malformed pdf file at link", pdf_link)
        return None