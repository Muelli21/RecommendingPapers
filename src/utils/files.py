import os
import sys
import json
import gzip

def object_size_gb(object):
    return sys.getsizeof(object) / 10**9

def save_json(file_path, object):
    """
    Saves a json file to the specified path

    Params: 
        file_path: specifies where to save the file
        json_file: python object to save
    """

    target_file = open(file_path, "w+")
    target_file.write(json.dumps(object))
    target_file.close()

def load_json(file_path):
    """
    Loads a json
    
    Params: 
        file_path: specifies the json file to load
    """
    
    try:
        with open(file_path, "r") as target_file:
            json_file = json.loads(target_file.read())
            
        target_file.close()
        return json_file
    except Exception as e:
        print("An exception occurred while trying to load the json file!")
        print(e)
        return {}

def save_txt(file_path, text):
    """
    Saves a txt file to the specified path

    Params: 
        file_path: specifies where to save the file
        text: text to save
    """

    target_file = open(file_path, "w+")
    target_file.write(text)
    target_file.close()

def load_txt(file_path):
    """
    Loads a txt file from the specified file path

    Params: 
    file_path: specifies the file to load
    """

    with open(file_path, "r") as target_file:
        return target_file.read()

def save_pdf(content, file_path):
    """
    Saves pdf to .pdf-file with specified `file_path`

    Params:
        content: 
        file_path:
    """

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(content)

def load_jsonl(file_path):
    """
    Loads each line of a gzip compressed file as a json object

    Returns: 
        Generator yielding one json object at a time
    """

    with open(file_path, 'r') as fin:
        for line in fin: 
            yield json.loads(line)

def load_compressed_jsonl(file_path):
    """
    Loads each line of a gzip compressed file as a json object

    Returns: 
        Generator yielding one json object at a time
    """

    with gzip.open(file_path, 'r') as fin:
        for line in fin: 
            yield json.loads(line)
