import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import pandas as pd

def min_max_normalize(arr):
    warnings.filterwarnings("error")
    try:
        min_val = np.min(arr)
        max_val = np.max(arr)
        return (arr - min_val) / (max_val - min_val)
    except:
        return arr

def z_score_normalize(arr):
    mu = np.nanmean(arr, axis=-1, keepdims=True)
    std = np.nanstd(arr, axis=-1, keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (arr - mu) / std



def verify_distribution_sr_af(df):
    print("Total sample count: ", len(df))
    print("Sex distribution:")
    sex_order = ['Female', 'Male']
    print(df['Sex'].value_counts(normalize=True).reindex(sex_order))
    print(df['Sex'].value_counts().reindex(sex_order))

    print("\nAge group distribution:")
    age_order = ['0-39', '40-59', '60-79', '80+']
    age_groups = pd.cut(df['Age'], bins=[0, 39, 59, 79, 1000], labels=age_order)
    print(age_groups.value_counts(normalize=True).reindex(age_order))
    print(age_groups.value_counts().reindex(age_order))

    print("\nDisease distribution:")
    print("AF:")
    print(df['AF'].value_counts(normalize=True).sort_index())
    print(df['AF'].value_counts().sort_index())
    print("SR:")
    print(df['SR'].value_counts(normalize=True).sort_index())
    print(df['SR'].value_counts().sort_index())

    print("\nSource distribution:")
    print(df['Source'].value_counts(normalize=True).sort_index())
    print(df['Source'].value_counts().sort_index())

    print("\nAge distribution for males and females with SR and AF:")
    for sex in ['Female', 'Male']:
        for condition, label in zip(['SR', 'AF'], ['Sinus Rhythm', 'Atrial Fibrillation']):
            subset_df = df[(df['Sex'] == sex) & (df[condition] == '1')]
            print(f"\n{sex} with {label}:")
            print("Total: ", len(subset_df))
            age_dist = pd.cut(subset_df['Age'], bins=[0, 39, 59, 79, 1000], labels=age_order)
            #print(age_dist.value_counts(normalize=True).reindex(age_order))
            print(age_dist.value_counts().reindex(age_order))

def verify_distribution_af_sr_mi_hyp(df):
    print("Total sample count: ", len(df))
    print("Sex distribution:")
    sex_order = ['Female', 'Male']
    print(df['Sex'].value_counts(normalize=True).reindex(sex_order))
    print(df['Sex'].value_counts().reindex(sex_order))

    print("\nAge group distribution:")
    age_order = ['0-39', '40-59', '60-79', '80+']
    age_groups = pd.cut(df['Age'], bins=[0, 39, 59, 79, 1000], labels=age_order)
    print(age_groups.value_counts(normalize=True).reindex(age_order))
    print(age_groups.value_counts().reindex(age_order))

    print("\nDisease distribution:")
    print("AF:")
    print(df['AF'].value_counts(normalize=True).sort_index())
    print(df['AF'].value_counts().sort_index())
    print("SR:")
    print(df['SR'].value_counts(normalize=True).sort_index())
    print(df['SR'].value_counts().sort_index())
    print("MI:")
    print(df['MI'].value_counts(normalize=True).sort_index())
    print(df['MI'].value_counts().sort_index())
    print("HYP:")
    print(df['HYP'].value_counts(normalize=True).sort_index())
    print(df['HYP'].value_counts().sort_index())

    print("\nSource distribution:")
    print(df['Source'].value_counts(normalize=True).sort_index())
    print(df['Source'].value_counts().sort_index())

    print("\nAge distribution for males and females with SR and AF:")
    for sex in ['Female', 'Male']:
        for condition, label in zip(['SR', 'AF', 'MI', 'HYP'], ['Sinus Rhythm', 'Atrial Fibrillation', 'MI', 'HYP']):
            subset_df = df[(df['Sex'] == sex) & (df[condition] == '1')]
            print(f"\n{sex} with {label}:")
            print("Total: ", len(subset_df))
            age_dist = pd.cut(subset_df['Age'], bins=[0, 39, 59, 79, 1000], labels=age_order)
            #print(age_dist.value_counts(normalize=True).reindex(age_order))
            print(age_dist.value_counts().reindex(age_order))

def balance_gender_af_sr(df, target_ratio=0.5, balance_set='test'):
    df = df.drop('Strata', axis='columns')
    df['Strata'] = df['AF'] + '_' + df['AgeGroup']
    male_df = df[df['Sex'] == 'Male']
    female_df = df[df['Sex'] == 'Female']

    # Determine the number of samples for each gender
    if balance_set == 'test':
        total_samples = min(len(male_df), len(female_df)) * 2
    else:
        total_samples = min(len(male_df), len(female_df))
    male_samples = int(total_samples * target_ratio)
    female_samples = total_samples - male_samples

    if male_samples != len(male_df) and target_ratio != 0:
        # Stratified sampling for males
        male_strata = male_df['Strata'].tolist()
        _, male_indices = train_test_split(
            range(len(male_df)),
            test_size=male_samples / len(male_df),
            stratify=male_strata,
            random_state=42
        )
        male_indices = male_df.index[male_indices].tolist()
    elif target_ratio == 0:
        male_indices = []
    else:
        male_indices = male_df.index.tolist()

    if female_samples != len(female_df) and target_ratio != 1:
        # Stratified sampling for females
        female_strata = female_df['Strata'].tolist()
        _, female_indices = train_test_split(
            range(len(female_df)),
            test_size=female_samples / len(female_df),
            stratify=female_strata,
            random_state=42
        )
        female_indices = female_df.index[female_indices].tolist()
    elif target_ratio == 1:
        female_indices = []
    else:
        female_indices = female_df.index.tolist()

    # Combine the indices
    balanced_indices = male_indices + female_indices
    return balanced_indices

