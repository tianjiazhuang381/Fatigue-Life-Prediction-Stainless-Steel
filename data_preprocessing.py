import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
def classify_stainless_steel(grade):
    g = str(grade)
    austenitic = ['301','302','303','304','305','308','309','310',
                  '316','317','321','347','348','SUS','S317','SCH','A286']
    martensitic = ['403','410','414','416','422','514']
    ferritic = ['430','446']
    ph = ['630','SUH']

    if any(g.startswith(p) for p in austenitic):
        return 'Austenitic'
    elif any(g.startswith(p) for p in martensitic):
        return 'Martensitic'
    elif any(g.startswith(p) for p in ferritic):
        return 'Ferritic'
    elif any(g.startswith(p) for p in ph):
        return 'PH'
    return 'Unknown'

def extract_fatigue_limit(df):
    fl_dict = {}
    for grade in df['Grade'].unique():
        sub = df[df['Grade'] == grade]
        idx = sub['Nf'].idxmax()
        fl_dict[grade] = sub.loc[idx, 'σa']
    return df['Grade'].map(fl_dict)

def compute_features(df, m=10):
    df['logN'] = np.log10(df['Nf'])
    df['Type'] = df['Grade'].apply(classify_stainless_steel)
    le = LabelEncoder()
    df['Type_enc'] = le.fit_transform(df['Type'])
    df['σfl'] = extract_fatigue_limit(df)
    df['σa/σuts'] = df['σa'] / df['σuts']
    df['σa/σys'] = df['σa'] / df['σys']
    df['G1'] = np.log10((df['σuts'] - df['σa']) / (df['σa'] - df['σfl'] + m))
    return df, le

def load_and_preprocess(filepath):
    df = pd.read_excel(filepath)
    df = df.dropna(axis=1, how='all')
    df.columns = ['Grade', 'σys', 'σuts', 'El', 'σa', 'Nf']
    df, le = compute_features(df)
    print(f"Dataset loaded: {len(df)} samples, {df['Grade'].nunique()} grades")
    print(f"Types: {dict(df['Type'].value_counts())}")
    return df, le

if __name__ == '__main__':
    df, le = load_and_preprocess('fatigue_dataset.xlsx')
    print("\nFeature correlations with log Nf:")
    for col in ['σys', 'σuts', 'El', 'σa', 'σfl', 'σa/σuts', 'σa/σys', 'G1']:
        r = df[col].corr(df['logN'])
        print(f"  {col:>8s}: r = {r:+.3f}")
    df.to_csv('processed_dataset.csv', index=False)
    print("\nSaved: processed_dataset.csv")