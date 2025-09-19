import pandas as pd

def is_consistent(hypothesis, instance):
    for h, x in zip(hypothesis, instance):
        if h != '?' and h != x:
            return False
    return True

def candidate_elimination(data):
    n_features = len(data[0]) - 1
    S = ['Ø'] * n_features  # Most specific hypothesis
    G = [['?'] * n_features]  # Most general hypothesis

    for row in data:
        instance, label = row[:-1], row[-1]
        if str(label).lower() == 'yes':
            # Remove inconsistent general hypotheses
            G = [g for g in G if is_consistent(g, instance)]

            # Update specific hypothesis S
            for i in range(n_features):
                if S[i] == 'Ø':
                    S[i] = instance[i]
                elif S[i] != instance[i]:
                    S[i] = '?'

        elif str(label).lower() == 'no':
            G_new = []
            for g in G:
                for i in range(n_features):
                    if g[i] == '?':
                        if S[i] != '?':
                            new_g = g.copy()
                            new_g[i] = S[i]
                            if is_consistent(new_g, instance) == False:
                                G_new.append(new_g)
            G = G_new

    return S, G

df =  [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]
S_final, G_final = candidate_elimination(df)

print("Final Specific Hypothesis (S):", S_final)
print("Final General Hypotheses (G):", G_final)
