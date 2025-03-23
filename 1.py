from utils import *

word_info_df, _, eyemovement_df = load_corpus('MECO')

# Проверка структуры данных
print("Columns in eyemovement_df:", eyemovement_df.columns)
print("Columns in word_info_df:", word_info_df.columns)

# Создание списков предложений и читателей
sn_list = np.unique(eyemovement_df.sentence_id.values).tolist()
reader_list = np.unique(eyemovement_df.reader_id.values).tolist()