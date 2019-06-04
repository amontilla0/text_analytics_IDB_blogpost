import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


df = pd.read_csv('activities.csv', sep='\t', header=0)

# adding main sector feature
df['MAIN_SECTOR'] = df.SECTOR.str[:2]

# taking as subset only 10 different categories
df = df[df['MAIN_SECTOR'].isin(['TR', 'IS', 'AS', 'EN', 'DU', 'FM', 'PA', 'ED', 'AG', 'SA'])]

#categories = ['soc.religion.christian', 'comp.graphics', 'sci.med', 'rec.sport.baseball']
#cat_names = ['Christianity', 'Computer Graphics', 'Medicine', 'Baseball']
#cat_map = dict(zip(categories, cat_names))
#twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
#
#print(type(df.as_matrix(columns=['OPER_NUM'])))
#
#print(np.array(df.SECTOR.tolist()))
#
#print(twenty_train.target)

le = LabelEncoder()
y = le.fit_transform(df.MAIN_SECTOR.tolist())
X = df.DESCRIPTION.tolist()

X_train, X_test, y_train, y_test  = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=4321,
        stratify=y)

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3
                     ,random_state=4321
                     , max_iter=20))])
text_clf.fit(X_train, y_train)
print("Model successfully trained.\n")

vc = df.MAIN_SECTOR.value_counts()

fig = plt.figure()
ax = fig.add_subplot(111)
cats = vc.index.tolist()
ax.bar(cats, vc.tolist())

#ax.set_xticks([0, 20, 40, 60, 80, 100, 120])

plt.title('Sector distribution')
plt.xlabel('Sector')
plt.ylabel('Count of projects')
plt.show()

vc = df.MAIN_SECTOR.value_counts().head(20)

fig = plt.figure()
ax = fig.add_subplot(111)
cats = vc.index.tolist()
ax.bar(cats, vc.tolist())

plt.title('Sector distribution (TOP 20)')
plt.xlabel('Sector')
plt.ylabel('Count of projects')
plt.show()

#eval_text = input("Enter some text to predict its category: ")
#pred_cat_idx = text_clf.predict([eval_text])[0]
#pred_cat = le.inverse_transform(pred_cat_idx)
#print("\n\nPREDICTION:\nYour text talks about {}.".format(pred_cat))

print('accuracy:', accuracy_score(y_test, text_clf.predict(X_test)))


# PARAMETER TUNING
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
'tfidf__use_idf': (True, False),
'clf__alpha': (1e-1, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, y_train)
print('\nbest score:', gs_clf.best_score_)
print('new accuracy:', accuracy_score(y_test, gs_clf.predict(X_test)))
for param_name in sorted(parameters.keys()):
  print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
