from config import *

cf = {"model_pretrained": "bert-base-multilingual-cased",
      "atten_type": 'local-g',
      "max_sn_len": 27,
      "max_sn_token": 35,
      "max_sp_len": 52}

DEVICE = "cpu"

dnn = model.Eyettention(cf)
dnn.eval()
state_dict = torch.load("../training_results/MECO/ML_ET.pth", map_location=torch.device('cpu'))
missing_keys, unexpected_keys = dnn.load_state_dict(state_dict, strict=False)
tokenizer = BertTokenizerFast.from_pretrained(cf['model_pretrained'])


english_sentences = open("../res/sentences_ml.txt", "r",encoding="UTF-8").readlines()

texts = []
for s in english_sentences:
    texts.append(('[CLS]' + ' ' + s + ' ' + '[SEP]').split())

tokens = [tokenizer(t, add_special_tokens=False, max_length=cf['max_sn_token'], padding='max_length', is_split_into_words=True) for t in texts]

word_ids = [t.word_ids() for t in tokens]
word_ids = [[val if val is not None else np.nan for val in wi] for wi in word_ids]

text_word_len = [[compute_word_length(txt) for txt in [text]] for text in texts]
text_word_len = [pad_seq(twl, cf['max_sn_len'], fill_value=np.nan, dtype=np.float32) for twl in text_word_len]

path = "../training_results/MECO/ML_FN.pickle"
with open(path, "rb") as file_to_read:
    loaded_dictionary = pickle.load(file_to_read)
sn_word_len_mean = loaded_dictionary['sn_word_len_mean']
sn_word_len_std = loaded_dictionary['sn_word_len_std']

text_word_len = [(twl - sn_word_len_mean) / sn_word_len_std for twl in text_word_len]
text_word_len = [np.nan_to_num(twl, nan=0) for twl in text_word_len]

for t in tokens:
    t['input_ids'] = torch.tensor([t['input_ids']]).to(DEVICE)
    t['attention_mask'] = torch.tensor([t['attention_mask']]).to(DEVICE)

word_ids = [torch.tensor([wi]).to(DEVICE) for wi in word_ids]
word_len = [torch.tensor([twl.squeeze()]).to(DEVICE) for twl in text_word_len]

le = LabelEncoder()
le.fit(np.append(np.arange(-cf["max_sn_len"] + 3, cf["max_sn_len"] - 1), cf["max_sn_len"] - 1))

scanpaths = [dnn.scanpath_generation(sn_emd=tokens[i]['input_ids'],
                                    sn_mask=tokens[i]['attention_mask'],
                                    word_ids_sn=word_ids[i],
                                    sn_word_len=word_len[i],
                                    le=le,
                                    max_pred_len=50) for i in range(len(english_sentences))]

scanpaths = [sp[0] for sp in scanpaths]
sn_lens = [(torch.max(torch.nan_to_num(wi), dim=1)[0] + 1 - 2).detach().to('cpu').numpy().tolist() for wi in word_ids]
scanpaths = [post_process_scanpath(x, y) for x, y in zip(scanpaths, sn_lens)]

human = [generate_sp(s)[0] for s in english_sentences]
human_sh = [generate_sp(s)[1] for s in english_sentences]

nld_model = []
nld_rand = []

for dnn_path, human_path in zip(scanpaths, human):
    nld = calculate_nld(dnn_path, human_path)
    nld_model.append(nld)

for dnn_path, human_path in zip(human, human_sh):
    nld = calculate_nld(dnn_path, human_path)
    nld_rand.append(nld)

mean_nld_model = np.mean(nld_model)
mean_nld_rand = np.mean(nld_rand)

if mean_nld_model < mean_nld_rand:
    print(f"Модель работает лучше, чем случайное предсказание. NLD предсказаний {mean_nld_model}, NLD перемешанной последовательности {mean_nld_rand}")
else:
    print(f"Модель работает на уровне случайного предсказания. {mean_nld_rand}")


plt.figure(figsize=(12, 6))
sns.kdeplot(human[1], label='Человеческие последовательности')
sns.kdeplot(human_sh[1], label='Перемешанные последовательности')
sns.kdeplot(scanpaths[1], label='Предсказанные последовательности')
plt.title('Distribution of Landing Positions')
plt.xlabel('Landing Position')
plt.ylabel('Density')
plt.legend()
plt.show()