import json

import requests
import warnings


from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver import Edge

from bs4 import BeautifulSoup

import utils.constants as constants
import utils.files as files
import preparation.extracting as extracting
import utils.parsing as parsing

HEADERS = {
    'user-agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:95.0) Gecko/20100101 Firefox/95.0",
    'content-type': 'application/pdf',
    'accept': '*/*',
    'accept-encoding': 'gzip, deflate, br',
}

CAPTCHA_STRING = "e=document.getElementById('captcha')"

def scrape_semantic_scholar_by_id(paper_id):
    """
    Scrapes a paper from semantic scholar based on its paper_id.

    Params: 
        paper_id: Semantic Scholar id of the paper that should be scraped

    Returns:
        scraped_paper: dictionary of scraped_paper (key: paper_id, value: details as abstract, year, authors etc.)
    """
    scraped_paper = {paper_id: {}}

    # abstract, authors, publishing year, reference count, citation count
    scraped_paper[paper_id] = {}
    paper_details_url = "https://api.semanticscholar.org/graph/v1/paper/%s?fields=title,abstract,authors,year,referenceCount,citationCount,fieldsOfStudy,s2FieldsOfStudy"
    paper_details_response = requests.get(paper_details_url % (paper_id))
    paper_details_dict = json.loads(paper_details_response.content.decode('utf-8'))
    scraped_paper[paper_id]['title'] = paper_details_dict['title']
    scraped_paper[paper_id]['abstract'] = paper_details_dict['abstract']
    scraped_paper[paper_id]['year'] = paper_details_dict['year']
    scraped_paper[paper_id]['authors'] = paper_details_dict['authors']
    scraped_paper[paper_id]['referenceCount'] = paper_details_dict['referenceCount']
    scraped_paper[paper_id]['citationCount'] = paper_details_dict['citationCount']
    scraped_paper[paper_id]['fieldsOfStudy'] = extracting.clean_s2orc_categories(paper_details_dict['fieldsOfStudy'])
    scraped_paper[paper_id]['s2FieldsOfStudy'] = paper_details_dict['s2FieldsOfStudy']

    # references
    paper_references_url = "https://api.semanticscholar.org/graph/v1/paper/%s/references"
    paper_references_response = requests.get(paper_references_url % (paper_id))
    paper_references_list = json.loads(paper_references_response.content.decode('utf-8'))['data']

    scraped_paper[paper_id]['referencesIds'] = []
    unresolved_reference_count = 0
    for reference in paper_references_list:
        if reference['citedPaper']['paperId'] is not None:
            scraped_paper[paper_id]['referencesIds'].append(reference['citedPaper']['paperId']) 
        else: 
            unresolved_reference_count += 1

    scraped_paper[paper_id]['unresolvedReferenceCount'] = unresolved_reference_count
    return scraped_paper

def scrape_semantic_scholar(search_query, num_results):
    """
    Scrapes links of .pdf-files from semantic scholar given a search query

    Params: 
        search_query: string
        num_results: number of results for which details are relevant
    
    Returns: 
        pdf_link_set: set of unique urls to .pdf-files
    """

    scraped_papers = {}

    # get results for search query
    search_query = search_query.lower().replace(" ", "+")
    search_query_url = "https://api.semanticscholar.org/graph/v1/paper/search?query=%s&limit=%d"
    search_query_response = requests.get(search_query_url % (search_query, num_results))
    search_query_results = json.loads(search_query_response.content.decode('utf-8'))['data']

    # get details for top n results 
    for paper_dict in search_query_results:
        result_id = paper_dict['paperId']
        scraped_paper = scrape_semantic_scholar_by_id(result_id)
        scraped_papers[result_id] = scraped_paper[result_id]

    return scraped_papers

def request_pdf(pdf_link:str):
    """
    Sends an HTTP-request to the passed `pdf_link` and returns the HTTP-response content.

    Params:
        pdf_link: Link to a .pdf-file that should be requested

    Returns:
        Content of HTTP-response for the pdf_link
    """

    response = None

    try:
        response = requests.get(pdf_link, headers = HEADERS)
        response.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        print ("Http Error:",errh)
    except requests.exceptions.ConnectionError as errc:
        print ("Error Connecting:",errc)
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)
    except requests.exceptions.RequestException as err:
        print ("OOps: Something Else",err)

    if (response is None or response.status_code != 200):
        if response is None:
            print("Retrieving the following pdf failed. No response has been received!")
        else: 
            print("Retrieving the following pdf failed with status code:", response.status_code)


        print("URL:", pdf_link)
        return None

    return response.content

def scrape_pdf_links(pdf_links, all_papers, pdfs_dir = constants.PDFS_DIR):
    """
    Scrapes the provided links of .pdf-files, parses them and stores them under their hash-value in the all_papers dict.
    Moreover, the original .pdf-files are stored in pdfs_dir

    Params:
        pdf_links: urls of .pdf-files
        all_papers: dict mapping hashes to json-representation of papers
        pdfs_dir: directory to store .pdf-files

    Returns:
        all_papers: dict with old and new key-value pairs
    """

    for pdf_link in pdf_links:

        print("[*] Parsing pdf-link: ", pdf_link)

        content = request_pdf(pdf_link)

        if content is not None: 
            properties = parsing.parse_pdf(content, pdf_link)

            if properties is not None:
                file_name = str(pdfs_dir / (properties["hash"] + ".pdf"))
                Hash = properties["hash"]
                all_papers[Hash] = properties

                files.save_pdf(content, file_name)
                print("[-] Stored pdf-link with hash ", Hash, "\n")
            else:
                print("Error: No properties found for ", pdf_link, "\n")
    
    return all_papers

def scrape_google_scholar(search_query, nr_pages):
    """
    Scrapes links of .pdf-files from google scholar given a search query

    Params: 
        search_query: string
        nr_pages: number of google scholar pages to be searched
    
    Returns: 
        pdf_link_set: set of unique urls to .pdf-files
    """

    pdf_link_set = set()

    search_query = search_query.lower().replace(" ", "+")
    searchLink = "https://scholar.google.com/scholar?start=%d&q=%s&hl=en&as_sdt=0,5"

    for page in range(0, nr_pages * 10, 10):
        # http-request to get response of search query
        response = requests.get(searchLink % (page, search_query), headers = HEADERS)
        # extracting html content from response
        html_content = response.text

        if CAPTCHA_STRING in html_content:
            warnings.warn("While scraping Google Scholar, a CAPTCHA occured! Further scraping was stopped!")
            return (pdf_link_set, True)

        # using BeautifulSoup to map html-tree-structure to interpretable elements
        soup = BeautifulSoup(html_content, features="lxml")

        for link_tag in soup.find_all('a'):
            # filter links that contain .pdf as suffix
            link = (link_tag.get("href"))

            if ".pdf" in link:
                # print extracted link
                # add link to set
                pdf_link_set.add(link)

    return (pdf_link_set, False)

def scrape_google_scholar_selenium(search_query, nr_pages):
    """
    Scrapes links of .pdf-files from google scholar given a search query

    Params: 
        search_query: string
        nr_pages: number of google scholar pages to be searched
    
    Returns: 
        pdf_link_set: set of unique urls to .pdf-files
    """

    pdf_link_set = set()

    search_query = search_query.lower().replace(" ", "+")
    searchLink = "https://scholar.google.com/scholar?start=%d&q=%s&hl=en&as_sdt=0,5"

    for page in range(0, nr_pages * 10, 10):

        driver = Edge(executable_path = "/Users/wal/Documents/msedgedriver", capabilities = {})
        driver.get(searchLink % (page, search_query))

        if CAPTCHA_STRING in driver.page_source:
            warnings.warn("While scraping Google Scholar, a CAPTCHA occured! Please solve it!")
            try:
                element_present = EC.presence_of_element_located((By.CLASS_NAME, 'gs_r'))
                WebDriverWait(driver, 60).until(element_present)
            except TimeoutException:
                print("Timed out waiting for page to load")
                return (pdf_link_set, True)

        soup = BeautifulSoup(driver.page_source, features="lxml")

        for link_tag in soup.find_all('a'):
            # filter links that contain .pdf as suffix
            link = (link_tag.get("href"))

            if ".pdf" in link:
                # print extracted link
                # add link to set
                pdf_link_set.add(link)

    return (pdf_link_set, False)
