import numpy as np
from hmmlearn import hmm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

states = ["Healthy", "Disease"]
n_states = len(states)
observations = ["Normal", "Mid", "Severe"]
n_observations = len(observations)

start_prob = np.array([0.7, 0.3])

trans_prob = np.array([
    [0.85, 0.15],
    [0.25, 0.75]
])
emission_prob = np.array([
    [0.85, 0.12, 0.03],
    [0.05, 0.35, 0.60]
])


model = hmm.CategoricalHMM(n_components=n_states, init_params="")
model.startprob_ = start_prob
model.transmat_ = trans_prob
model.emissionprob_ = emission_prob

X, Z = model.sample(100)  # X: observations, Z: hidden states
y_true = Z.ravel()
y_obs = X.ravel()

y_pred = model.predict(X)

logprob, y_pred_viterbi = model.decode(X, algorithm="viterbi")

print("Accuracy:",accuracy_score(y_true,y_pred))
print("Precision:",precision_score(y_true,y_pred))
print("Recall:",recall_score(y_true,y_pred))
print("F1 Score:",f1_score(y_true,y_pred))
print("classification Report:",classification_report(y_true,y_pred))

print("\nSample Diagnosis:")
for i in range(10):
    print(f"Symptom: {observations[y_obs[i]]}, Predicted state: {states[y_pred[i]]}, True state: {states[y_true[i]]}")



'''
output:

Accuracy: 0.92
Precision: 0.9
Recall: 0.9
F1 Score: 0.9
classification Report:               precision    recall  f1-score   support

           0       0.93      0.93      0.93        60
           1       0.90      0.90      0.90        40

    accuracy                           0.92       100
   macro avg       0.92      0.92      0.92       100
weighted avg       0.92      0.92      0.92       100


Sample Diagnosis:
Symptom: Normal, Predicted state: Healthy, True state: Healthy
Symptom: Normal, Predicted state: Healthy, True state: Healthy
Symptom: Normal, Predicted state: Healthy, True state: Healthy
Symptom: Mid, Predicted state: Healthy, True state: Healthy
Symptom: Normal, Predicted state: Healthy, True state: Healthy
Symptom: Normal, Predicted state: Healthy, True state: Healthy
Symptom: Mid, Predicted state: Healthy, True state: Healthy
Symptom: Normal, Predicted state: Healthy, True state: Healthy
Symptom: Normal, Predicted state: Healthy, True state: Healthy
Symptom: Normal, Predicted state: Healthy, True state: Healthy
'''
