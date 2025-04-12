import pandas as pd
import numpy as np

try:
    with open('sentences.csv', 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    sentences_df = pd.DataFrame({'sentence': sentences, 'line_num': range(1, len(sentences)+1)})
    print(f"Загружено предложений: {len(sentences_df)}")
except Exception as e:
    print(f"Ошибка чтения sentences.csv: {str(e)}")
    exit()

try:
    sentences_tags_df = pd.read_csv('sentences_tags.csv')
    print(f"Загружено метаданных предложений: {len(sentences_tags_df)}")
    
    if len(sentences_df) != len(sentences_tags_df):
        print(f"Предупреждение: Количество предложений ({len(sentences_df)}) не совпадает с количеством метаданных ({len(sentences_tags_df)})")
        
        sentences_df['trialid'] = sentences_tags_df['trialid'].iloc[:len(sentences_df)]
except Exception as e:
    print(f"Ошибка чтения sentences_tags.csv: {str(e)}")
    exit()

word_info_data = []
for idx, row in sentences_tags_df.iterrows():
    try:
        if idx >= len(sentences_df):
            continue
            
        sentence = sentences_df.iloc[idx]['sentence']
        words = sentence.split()
        for word_pos, word in enumerate(words, start=1):
            word_info_data.append({
                'sentnum': row['trialid'],
                'word': word,
                'word_position': word_pos,
                'word_length': len(word)
            })
    except Exception as e:
        print(f"Ошибка обработки предложения {idx+1}: {str(e)}")

word_info_df = pd.DataFrame(word_info_data)
word_info_df.to_csv('word_info.csv', index=False)
print(f"Создано записей о словах: {len(word_info_df)}")

try:
    fixations_df = pd.read_csv('fixations.csv', 
                              dtype={'sentnum': 'float64', 'wordnum': 'float64'},
                              low_memory=False)
    
    fixations_df = fixations_df.loc[:, ~fixations_df.columns.duplicated()]
    
    eyemovement_df = fixations_df.rename(columns={
        'trialid': 'sentnum',
        'wordnum': 'word_position',
        'xn': 'landing_pos_norm',
        'dur': 'fixation_dur'
    })[['subid', 'sentnum', 'word_position', 'landing_pos_norm', 'fixation_dur']].copy()
    
    eyemovement_df = eyemovement_df[
        (eyemovement_df['fixation_dur'] > 50) & 
        (eyemovement_df['word_position'] > 0) &
        (eyemovement_df['landing_pos_norm'].notna())
    ].drop_duplicates()
    
    eyemovement_df['landing_pos_norm'] = pd.to_numeric(eyemovement_df['landing_pos_norm'], errors='coerce').clip(0, 1)
    eyemovement_df['sentnum'] = eyemovement_df['sentnum'].astype('float64')
    eyemovement_df['word_position'] = eyemovement_df['word_position'].astype('int64')
    
    eyemovement_df.to_csv('eyemovement.csv', index=False)
    print(f"Создано записей о фиксациях: {len(eyemovement_df)}")
    
except Exception as e:
    print(f"Ошибка обработки fixations.csv: {str(e)}")
    exit()
print("\nПроверка целостности данных:")
print(f"Уникальных предложений в word_info: {word_info_df['sentnum'].nunique()}")
print(f"Уникальных предложений в eyemovement: {eyemovement_df['sentnum'].nunique()}")

try:
    merged = pd.merge(
        eyemovement_df.drop_duplicates(['sentnum', 'word_position']),
        word_info_df.drop_duplicates(['sentnum', 'word_position']),
        on=['sentnum', 'word_position'],
        how='left'
    )
    
    missing_words = merged['word'].isna().sum()
    if missing_words > 0:
        print(f"Предупреждение: {missing_words} фиксаций ссылаются на отсутствующие слова")
        print("Примеры проблемных записей:")
        print(merged[merged['word'].isna()].head())
    
    print("Преобразование данных завершено успешно!")
    
except Exception as e:
    print(f"Ошибка при проверке целостности: {str(e)}")