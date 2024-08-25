import pandas as pd

# 读取CSV文件
data1 = pd.read_csv("KDDTrain.csv")
data2 = pd.read_csv("KDDTest.csv")
data = pd.concat([data1, data2])
unique_labels = data['label'].unique()
print(unique_labels)
attack_type_to_label = {
    # 5:normal 1:dos 2:probe 3:u2r 4:r2l
    'normal': 5,
    'neptune': 1,
    'warezclient': 4,
    'ipsweep': 2,
    'portsweep': 2,
    'teardrop': 1,
    'nmap': 2,
    'satan': 2,
    'smurf': 1,
    'pod': 1,
    'back': 1,
    'guess_passwd': 4,
    'ftp_write': 4,
    'multihop': 4,
    'rootkit': 3,
    'buffer_overflow': 3,
    'imap': 4,
    'warezmaster': 4,
    'phf': 4,
    'land': 1,
    'loadmodule': 3,
    'spy': 4,
    'perl': 3,
    'saint': 2,
    'mscan': 2,
    'apache2': 1,
    'snmpgetattack': 4,
    'processtable': 1,
    'httptunnel': 4,
    'ps': 3,
    'snmpguess': 4,
    'mailbomb': 1,
    'named': 4,
    'sendmail': 4,
    'xterm': 3,
    'worm': 1,
    'xlock': 4,
    'xsnoop': 4,
    'sqlattack': 3,
    'udpstorm': 1
}


data1["label"] = data1["label"].map(attack_type_to_label).astype(int)
data1.to_csv("KDDTrain5.csv", index=False)
data2["label"] = data2["label"].map(attack_type_to_label).astype(int)
data2.to_csv("KDDTest5.csv", index=False)
