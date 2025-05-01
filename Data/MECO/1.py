import pandas as pd

# Чтение CSV-файла
df = pd.read_csv('eyemovement.csv')

# # Удаление строк, где первый столбец содержит 'du_' в начале
# df = df[~df.iloc[:, 0].astype(str).str.startswith('du_')]

# # Сохранение обратно в CSV
# df.to_csv('eyemovement_no_du.csv', index=False)

# Оставить только строки, где первый столбец содержит 'ru_'
df = df[df.iloc[:, 0].astype(str).str.contains('ru_', case=False, na=False)]

# Сохранение в новый файл
df.to_csv('eyemovement_ru.csv', index=False)