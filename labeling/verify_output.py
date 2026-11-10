# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv(r'd:\btl kpdl\labeling\facebook_comments_labeled.csv', encoding='utf-8-sig')
print('=== LABELED DATASET INFO ===')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('Label distribution:')
print(df['label'].value_counts(dropna=False))
print('Needs review count:')
print(df['needs_review'].value_counts(dropna=False))
print('Average confidence:', round(df['confidence'].mean(), 4))

df_rev = pd.read_csv(r'd:\btl kpdl\labeling\facebook_comments_review.csv', encoding='utf-8-sig')
print('\n=== REVIEW DATASET INFO ===')
print('Shape:', df_rev.shape)

