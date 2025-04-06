from utils import *

word_info_df, _, eyemovement_df = load_corpus("MECO")
print("Столбцы в eyemovement_df:", eyemovement_df.columns.tolist())
print("Столбцы в word_info_df:", word_info_df.columns.tolist())

