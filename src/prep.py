import pandas as pd 
import numpy as np 
import os

# nu s 0/1
non_bin=["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]

def load(filepath):
    if not os.path.exists(filepath):
        print("Invalid path for the csv")
        return None
    print("aici")
    return pd.read_csv(filepath)    

# vreau sa dau 0/1 dupa mediana din coloana respectiva 
# asta ramana up for testing totusi ca nu stiu daca e optim asa
def to_bin(data):
    x=data.copy()
    for col in non_bin:
        thresh=x[col].median()
        x[col]=(x[col]>=thresh).astype(int)
    return x.astype(int)

def split_data(data, seed, res="Diabetes_binary", test_prc=0.2):
    np.random.seed(seed)
    perm=np.random.permutation(len(data))
    test_cnt=int(len(data)*test_prc)

    train_data=data.iloc[perm[:test_cnt]]
    test_data=data.iloc[perm[test_cnt:]]

    not_res=[col for col in data.columns if col!=res]

    res_train=train_data[res].values 
    res_test=test_data[res].values
    d_train=train_data[not_res].values
    d_test=test_data[not_res].values

    return d_train, res_train, d_test, res_test, not_res


def pre(seed, filepath="./data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv", res="Diabetes_binary", test_prc=0.2):
    data=load(filepath)
    return split_data(data, seed, res, test_prc)