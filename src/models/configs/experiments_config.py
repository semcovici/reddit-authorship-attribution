# External
import sys 
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

estimators = [
        ('svm', LinearSVC(random_state=42, max_iter=10000)),
        ('lr_l1', LogisticRegression(random_state=42, penalty="l1", solver="liblinear"),
        ('rf'), RandomForestClassifier(random_state=42))]
stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=42, penalty="l2", solver="liblinear"))

clfs = [MultinomialNB(),
        LogisticRegression(random_state=42, penalty="l1", solver="liblinear"),
        LogisticRegression(random_state=42, penalty="l2", solver="liblinear"),
        LinearSVC(random_state=42, max_iter=10000),
        SVC(random_state=42),
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(random_state=42),
        AdaBoostClassifier(random_state=42),
        GradientBoostingClassifier(random_state=42),
        stacking
        ]

vectorizers = [CountVectorizer(ngram_range=(1,1), analyzer="word"), 
                CountVectorizer(ngram_range=(1,3), analyzer="word"), 
                TfidfVectorizer(ngram_range=(1,1), analyzer="word"), 
                TfidfVectorizer(ngram_range=(1,3), analyzer="word")]