from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

docs = [
    "Machine Learning improves Artificial Intelligence",
    "Football is a popular Outdoor Sport",
    "Deep Learning is a Branch of Machine Learning",
    "Cricket is a Land Loved by Many People",
    "Neural Networks help in AI Appllications",
    "Tennis is a Olympic Sport"
]

labels = [1,0,1,0,1,0]
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(docs)
x_train,x_test,y_train,y_test = train_test_split(x,labels,test_size=0.3,random_state=42)
model = SVC()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Precision:",precision_score(y_test,y_pred))
print("Recall:",recall_score(y_test,y_pred))
print("F1 Score:",f1_score(y_test,y_pred))
print("Actual labels",y_test)
print("Predicted labels",y_pred)



#output:

Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1 Score: 1.0
Actual labels [1, 0]
Predicted labels [1 0]
