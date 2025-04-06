import numpy as np
import model
import torch
from torch.utils import model_zoo
from transformers import BertTokenizerFast
import pickle
from sklearn.preprocessing import LabelEncoder
import Levenshtein
import matplotlib.pyplot as plt
import seaborn as sns
from LAC import LAC

def compute_word_length(txt):
    if isinstance(txt, str):
        word_lengths = [len(word) for word in txt.split()]
    elif isinstance(txt, (list, np.ndarray)) and all(isinstance(x, str) for x in txt):
        word_lengths = [len(word) for word in txt]
    elif isinstance(txt, (list, np.ndarray)) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in txt):
        word_lengths = list(txt)
    else:
        raise ValueError("Неподдерживаемый тип входных данных. Ожидается строка, список слов или массив длин слов.")
    
    word_lengths = [np.nan] + word_lengths + [np.nan]
    arr = np.array(word_lengths, dtype=np.float64)
    
    arr[arr == 0] = 1/(0 + 0.5)
    arr[arr != 0] = 1/arr[arr != 0]
    
    return arr

def pad_seq(seqs, max_len, dtype=np.int32, fill_value=np.nan):
    padded = np.full((len(seqs), max_len), fill_value=fill_value, dtype=dtype)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
    return padded

def post_process_scanpath(syn_scanpath, sn_len):
    syn_scanpath = syn_scanpath.detach().to('cpu').numpy()
    max_sp_len = syn_scanpath.shape[1]

    stop_indx = []
    for i in range(syn_scanpath.shape[0]):
        stop = np.where(syn_scanpath[i,:]==(sn_len[i]+1))[0]
        if len(stop)==0:
            stop_indx.append(max_sp_len-1)
        else:
            stop_indx.append(stop[0])

    syn_scanpath_cut = [syn_scanpath[i][1:stop_indx[i]] for i in range(syn_scanpath.shape[0])]
    return syn_scanpath_cut

def flatten_list(nested_list):

    flat_list = []
    for item in nested_list:
        if isinstance(item, (list, np.ndarray)):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def calculate_nld(seq1, seq2):

    if isinstance(seq1, np.ndarray):
        seq1 = seq1.tolist()
    if isinstance(seq2, np.ndarray):
        seq2 = seq2.tolist()

    seq1 = flatten_list(seq1)
    seq2 = flatten_list(seq2)

    seq1 = [-1 if np.isnan(x) else x for x in seq1]
    seq2 = [-1 if np.isnan(x) else x for x in seq2]

    ld = Levenshtein.distance(seq1, seq2)

    max_len = max(len(seq1), len(seq2))
    return ld / max_len if max_len > 0 else 0

def generate_sp(sentence):
    true_sp = list(range(len(sentence.split())))
    return (true_sp, np.random.permutation(true_sp).tolist())

def compute_BSC_word_length(sentence, lac):
    word_string = lac.run(sentence)
    #print(word_string)
    word_len = [len(i) for i in word_string]
    wl_list = []
    for wl in word_len:
        wl_list.extend([wl]*wl)
    arr = np.asarray(wl_list, dtype=np.float32)
    #length of a punctuation is 0, plus an epsilon to avoid division output inf
    arr[arr==0] = 1/(0+0.5)
    arr[arr!=0] = 1/(arr[arr!=0])
    return arr