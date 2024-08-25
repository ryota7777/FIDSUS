import pandas as pd

# 读取CSV文件
data1 = pd.read_csv("unsw_train.csv")
data2 = pd.read_csv("unsw_test.csv")

# print(unique1)
# print(unique2)
#  ['Normal' 'Backdoor' 'Analysis' 'Fuzzers' 'Shellcode' 'Reconnaissance'
#  'Exploits' 'DoS' 'Worms' 'Generic']
attack_type_to_label = {
    "Normal": 1,
    "Backdoor": 2,
    "Analysis": 3,
    "Fuzzers": 4,
    "Shellcode": 5,
    "Reconnaissance": 6,
    "Exploits": 7,
    "DoS": 8,
    "Worms": 9,
    "Generic": 10
}

data1["label"] = data1["attack_cat"].map(attack_type_to_label).astype(int)
data1.to_csv("unsw_train10.csv", index=False)
data2["label"] = data2["attack_cat"].map(attack_type_to_label).astype(int)
data2.to_csv("unsw_test10.csv", index=False)

