import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from fairseq.models.bart import BARTModel

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

BART_ARGS = {
    "num_beams": 4,
    "length_penalty": 2,
    "max_length": 142,
    "min_length": 56,
    "no_repeat_ngram_size": 3
}

SCITLDR_ARGS = {
    "num_beams": 2,
    "length_penalty": 0.4,
    "max_length": 30,
    "min_length": 5,
    "no_repeat_ngram_size": 3
}

def summarize(paper_id, papers, model = "BART", content = "content"):
    """
    Summarizes a single paper specified by its hash

    Params: 
        paper_hash: hash to identify the paper
        papers: dict of papers containing the paper
        model: language model used (options: BART, SciTLDR)
        content: key specifying the field of the paper dict to be used (e.g. abstract, content...)
    
    Returns: 
        summary string
    """

    text = papers[paper_id][content]
    text = text.replace('\n','')

    return summarize_batch([text], model)

def summarize_batch(texts, model = "BART"):
    """
    Summarizes a provided batch of texts

    Params: 
        texts: list containing texts to be summarized
        model: language model used (options: BART, SciTLDR)
    
    Returns: 
        summarization as string
    """

    if model == "BART":
        return bart_summarize(texts, **BART_ARGS)

    if model == "SciTLDR":
        return sciTLDR_summarize(texts, **SCITLDR_ARGS)

def bart_summarize(texts, num_beams, length_penalty, max_length, min_length, no_repeat_ngram_size):
    """
    Summarizes a provied batch of texts using the BART model: 
    https://arxiv.org/abs/1910.13461

    Params: 
        texts: list containing texts to be summarized
        num_beams:
        length_penalty:
        max_length:
        min_length:
        no_repeat_ngram_size: 
    
    Returns: 
        list containing summary strings
    """

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    text_input_ids = tokenizer.batch_encode_plus(texts, return_tensors='pt', max_length=1024, truncation=True, padding=True)['input_ids'].to(torch_device)
    summary_ids = model.generate(text_input_ids, num_beams=num_beams, length_penalty=length_penalty, max_length=max_length, min_length=min_length, no_repeat_ngram_size=no_repeat_ngram_size)    
    summary_txt = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    return summary_txt

def sciTLDR_summarize(texts, num_beams, length_penalty, max_length, min_length, no_repeat_ngram_size):
    """
    Summarizes a provied batch of texts using the SciTLDR model:
    https://arxiv.org/abs/2004.15011

    Params: 
        texts: list containing texts to be summarized
        num_beams:
        length_penalty:
        max_length:
        min_length:
        no_repeat_ngram_size: 
    
    Returns: 
        list containing summary strings
    """

    model_name_or_path = "../data/models/SciTLDR/"
    data_name_or_path = "./SciTLDR-Data/SciTLDR-A/ctrl"

    bart = BARTModel.from_pretrained(
        model_name_or_path = model_name_or_path,
        data_name_or_path = data_name_or_path + '-bin',
    ).to(torch_device)

    summary_txt = bart.sample(texts, beam=num_beams, lenpen=length_penalty, max_len_b=max_length, min_len=min_length, no_repeat_ngram_size=no_repeat_ngram_size)
    return summary_txt