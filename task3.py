import pandas as pd, math
from collections import Counter

data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/play_tennis.csv")

def entropy(col):
    c, t = Counter(col), len(col)
    return -sum((v/t)*math.log2(v/t) for v in c.values())

def info_gain(df, attr, target):
    t_entropy = entropy(df[target])
    w_entropy = sum((len(sub)/len(df))*entropy(sub[target]) for _, sub in df.groupby(attr))
    return t_entropy - w_entropy

def id3(df, target, attrs):
    if len(set(df[target])) == 1: return df[target].iloc[0]
    if not attrs: return Counter(df[target]).most_common(1)[0][0]
    best = max(attrs, key=lambda a: info_gain(df, a, target))
    tree = {best: {}}
    for v, sub in df.groupby(best):
        tree[best][v] = id3(sub, target, [a for a in attrs if a != best])
    return tree

def predict(tree, sample):
    if not isinstance(tree, dict): return tree
    root = next(iter(tree))
    return predict(tree[root].get(sample.get(root), "Unknown"), sample)

attrs = [c for c in data.columns if c != "PlayTennis"]
tree = id3(data, "PlayTennis", attrs)
print("Decision Tree:", tree)

sample = {"Outlook": "Sunny","Temperature":"Cool","Humidity":"High","Wind":"Strong"}
print("Classification:", predict(tree, sample))






//Input :Outlook,Temperature,Humidity,Wind,PlayTennis
Sunny,Hot,High,Weak,No
Sunny,Hot,High,Strong,No
Overcast,Hot,High,Weak,Yes
Rain,Mild,High,Weak,Yes
Rain,Cool,Normal,Weak,Yes
Rain,Cool,Normal,Strong,No
Overcast,Cool,Normal,Strong,Yes
Sunny,Mild,High,Weak,No
Sunny,Cool,Normal,Weak,Yes
Rain,Mild,Normal,Weak,Yes
Sunny,Mild,Normal,Strong,Yes
Overcast,Mild,High,Strong,Yes
Overcast,Hot,Normal,Weak,Yes
Rain,Mild,High,Strong,No







