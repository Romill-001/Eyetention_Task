from utils import *
from config import *

e = pd.read_csv('./Data/MECO/eyemovement.csv')
#e['subid'] = (e['subid'].str.replace(r'\D+', '', regex=True).astype('Int64'))
e['subid'] = pd.to_numeric(
    e['subid'].str.replace(r'\D+', '', regex=True), 
    errors='coerce'
)

print(e)