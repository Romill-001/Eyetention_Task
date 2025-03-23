from utils import *

word_info_df, _, eyemovement_df = load_corpus('MECO')

# Создание списков предложений и читателей
sn_list = np.unique(eyemovement_df.sentnum.values).tolist()  # Используем sentnum
reader_list = np.unique(eyemovement_df.subid.values).tolist()  # Используем subid