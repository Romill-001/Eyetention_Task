from config import *

def ml_eng_output(cf, sentence):
    DEVICE = 'cpu'
    dnn = model.Eyettention(cf)
    dnn.eval()
    state_dict = torch.load(cf["et_path"], map_location=torch.device('cpu'))
    missing_keys, unexpected_keys = dnn.load_state_dict(state_dict, strict=False)
    tokenizer = BertTokenizerFast.from_pretrained(cf['model_pretrained'])
    text = ('[CLS]' + ' ' + sentence + ' ' + '[SEP]').split()
    tokens = tokenizer(text, add_special_tokens=False, max_length=cf['max_sn_token'], padding='max_length', is_split_into_words=True)
    word_ids = tokens.word_ids()
    word_ids = [val if val is not None else np.nan for val in word_ids]
    text_word_len = [compute_word_length(txt) for txt in [text]]
    text_word_len = pad_seq(text_word_len, cf['max_sn_len'], fill_value=np.nan, dtype=np.float32)
    path = cf["fn_path"]
    with open(path, "rb") as file_to_read:
        loaded_dictionary = pickle.load(file_to_read)
    sn_word_len_mean = loaded_dictionary['sn_word_len_mean']
    sn_word_len_std = loaded_dictionary['sn_word_len_std']
    text_word_len = (text_word_len - sn_word_len_mean)/sn_word_len_std
    text_word_len = np.nan_to_num(text_word_len)
    tokens['input_ids'] = torch.tensor([tokens['input_ids']]).to(DEVICE)
    tokens['attention_mask'] = torch.tensor([tokens['attention_mask']]).to(DEVICE)
    word_ids = torch.tensor([word_ids]).to(DEVICE)
    word_len = torch.tensor([text_word_len.squeeze()]).to(DEVICE)
    le = LabelEncoder()
    le.fit(np.append(np.arange(-cf["max_sn_len"]+3, cf["max_sn_len"]-1), cf["max_sn_len"]-1))
    syn_scanpath, density_pred = dnn.scanpath_generation(sn_emd=tokens['input_ids'],
                                                     sn_mask=tokens['attention_mask'],
                                                     word_ids_sn=word_ids,
                                                     sn_word_len = word_len,
                                                     le=le,
                                                     max_pred_len=50)
    sn_len = (torch.max(torch.nan_to_num(word_ids), dim=1)[0]+1-2).detach().to('cpu').numpy()
    syn_scanpath = post_process_scanpath(syn_scanpath, sn_len)
    fixated_word = [text[idx] for idx in syn_scanpath[0]]
    return syn_scanpath, density_pred, fixated_word

def chn_output(cf, sentence):
    DEVICE = 'cpu'
    dnn = model.Eyettention(cf)
    dnn.eval()
    state_dict = torch.load(cf["et_path"], map_location=torch.device('cpu'))
    missing_keys, unexpected_keys = dnn.load_state_dict(state_dict, strict=False)
    tokenizer = BertTokenizerFast.from_pretrained(cf['model_pretrained'])
    # text = ('[CLS]' + ' ' + sentence + ' ' + '[SEP]').split()
    tokens = tokenizer(sentence, add_special_tokens=True, max_length=cf['max_sn_len'], padding='max_length')
    lac = LAC(mode="seg")
    text_word_len = [compute_BSC_word_length(txt, lac) for txt in [sentence]]
    text_word_len = pad_seq(text_word_len, cf['max_sn_len'], fill_value=np.nan, dtype=np.float32)
    path = cf["fn_path"]
    with open(path, "rb") as file_to_read:
        loaded_dictionary = pickle.load(file_to_read)
    sn_word_len_mean = loaded_dictionary['sn_word_len_mean'].numpy()
    sn_word_len_std = loaded_dictionary['sn_word_len_std'].numpy()
    text_word_len = (text_word_len - sn_word_len_mean)/sn_word_len_std
    text_word_len = np.nan_to_num(text_word_len)
    tokens['input_ids'] = torch.tensor([tokens['input_ids']]).to(DEVICE)
    tokens['attention_mask'] = torch.tensor([tokens['attention_mask']]).to(DEVICE)
    word_len = torch.tensor([text_word_len.squeeze()]).to(DEVICE)
    le = LabelEncoder()
    le.fit(np.append(np.arange(-cf["max_sn_len"]+3, cf["max_sn_len"]-1), cf["max_sn_len"]-1))
    syn_scanpath, density_pred = dnn.scanpath_generation(sn_emd=tokens['input_ids'],
                                                        sn_mask=tokens['attention_mask'],
                                                        word_ids_sn=None,
                                                        sn_word_len = word_len,
                                                        le=le,
                                                        max_pred_len=50)
    sn_len = (torch.sum(tokens['attention_mask'], axis=1) - 2).detach().to('cpu').numpy()
    syn_scanpath = post_process_scanpath(syn_scanpath, sn_len)
    fixated_character = [sentence[idx-1] for idx in syn_scanpath[0]]
    return syn_scanpath, density_pred, fixated_character









