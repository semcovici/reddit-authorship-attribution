import sys
sys.path.append('../../src/')
from dataset.data_utils import SparseToArray

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

from imblearn.pipeline import Pipeline as ImbPipeline
class AuthorClassifier:
    def __init__(self, 
                vectorizer=CountVectorizer(), 
                clf=MultinomialNB(),
                embeddings=False,
                scaler=None,
                pca=None,
                kbest=None,
                param_grid=None,
                sampling = None
                ):
        self.vectorizer = vectorizer
        self.clf = clf
        self.embeddings = embeddings
        self.scaler = scaler
        self.pca = pca
        self.kbest = kbest
        self.param_grid = param_grid
        self.pipe = None
        self.predict_proba = None
        self.sampling = sampling

    def fit(self, X_train, y_train, params=[]):
        steps = list() 
        

        if not(self.embeddings): 

            steps.append(("vectorizer", self.vectorizer))
            steps.append(("sampling", self.sampling))
            
            if (self.scaler or self.pca) and not(isinstance(self.scaler, MaxAbsScaler)): 
                steps.append(("SparseToArray()", SparseToArray()))
            
            if self.scaler: 
                steps.append(("scaler", self.scaler))
            if self.pca: 
                steps.append(("pca", self.pca))
            if self.kbest:
                steps.append(("kbest", self.kbest))
        
        elif self.scaler: 
            steps.append(("sampling", self.sampling))
            steps.append(("scaler", self.scaler))

        steps.append(("clf", self.clf))
        pipe = ImbPipeline(steps)
                
        if self.param_grid:
            pipe = GridSearchCV(pipe, self.param_grid, n_jobs=-1, cv=3)
                    
        pipe.fit(X_train, y_train)
        
        self.pipe = pipe
        return self.pipe
    
    def predict(self, X_test):
        y_pred = self.pipe.predict(X_test)
        if isinstance(self.pipe["clf"], LinearSVC) or isinstance(self.pipe["clf"], SVC): 
            pred_probability = []
            for eachArr in self.pipe.decision_function(X_test):
                pred_probability.append(softmax(eachArr))
            self.predict_proba = pred_probability
        else:
            self.predict_proba = self.pipe.predict_proba(X_test)
        return y_pred
    
    def get_best_params(self):
        if self.pipe: return self.pipe.best_params_
        return None

    def get_best_estimator(self):
        if self.pipe: return self.pipe.best_estimator_
        return None
    
    # def evaluate(self, y_true, y_pred):
    #     metrics = dict()
    #     for i, author in enumerate(np.unique(y_true)):
    #         i += 1
    #         metrics[f"author{i}"] = author
    #     for i, author in enumerate(np.unique(y_true)):
    #         i += 1
    #         metrics[f"precision_author{i}"] = round(precision_score(y_true, y_pred, pos_label=author), 4)
    #         metrics[f"recall_author{i}"] = round(recall_score(y_true, y_pred, pos_label=author), 4)
    #         metrics[f"f1_score_author{i}"] = round(f1_score(y_true, y_pred, pos_label=author), 4)
    #     metrics["precision_weighted"] = round(precision_score(y_true, y_pred, average='weighted'), 4 )
    #     metrics["precision_micro"] = round(precision_score(y_true, y_pred, average='micro'), 4 )
    #     metrics["precision_macro"] = round(precision_score(y_true, y_pred, average='macro'), 4 )
    #     metrics["recall_weighted"] = round(recall_score(y_true, y_pred, average='weighted'), 4 )
    #     metrics["recall_micro"] = round(recall_score(y_true, y_pred, average='micro'), 4 )
    #     metrics["recall_macro"] = round(recall_score(y_true, y_pred, average='macro'), 4 )
    #     metrics["f1_weighted"] = round(f1_score(y_true, y_pred, average='weighted'), 4 )
    #     metrics["f1_micro"] = round(f1_score(y_true, y_pred, average='micro'), 4 )
    #     metrics["f1_macro"] = round(f1_score(y_true, y_pred, average='macro'), 4 )
    #     metrics["auc_score"] = round(roc_auc_score(y_true, self.predict_proba[:,1]), 4)
    #     metrics["accuracy"] = round(accuracy_score(y_true, y_pred), 4 )

    #     return metrics
    
    def evaluate(self, y_true, y_pred):

        metrics = dict()
        
        for i, author in enumerate(np.unique(y_true)):
            i += 1
            metrics[f"author{i}"] = author
        
        if len(np.unique(y_true)) == 2:
            
            
            
            for i, author in enumerate(np.unique(y_true)):
                i += 1
                metrics[f"precision_author{i}"] = round(precision_score(y_true, y_pred, pos_label=author), 4)
                metrics[f"recall_author{i}"] = round(recall_score(y_true, y_pred, pos_label=author), 4)
                metrics[f"f1_score_author{i}"] = round(f1_score(y_true, y_pred, pos_label=author), 4)
            
            
            if isinstance(self.pipe["clf"], LinearSVC) or isinstance(self.pipe["clf"], SVC): 
                metrics["auc_score"] = round(roc_auc_score(y_true, self.predict_proba), 4)
            else:
                metrics["auc_score"] = round(roc_auc_score(y_true, self.predict_proba[:,1]), 4)
                    
        else:
            
            for i, author in enumerate(np.unique(y_true)):
                i += 1
                metrics[f"precision_author{i}"] = round(precision_score(y_true, y_pred, average=None, labels=[author])[0], 4)
                metrics[f"recall_author{i}"] = round(recall_score(y_true, y_pred, average=None, labels=[author])[0], 4)
                metrics[f"f1_score_author{i}"] = round(f1_score(y_true, y_pred, average=None, labels=[author])[0], 4)
            
            
            pd.DataFrame({
                'a': y_true,
                'b': y_pred,
                'c': str(self.pipe["clf"])
            }).to_csv('teste.csv')
            
            np.save('teste',self.predict_proba)
            
            metrics["auc_score_ovr"] = round(roc_auc_score(y_true, self.predict_proba,multi_class='ovr' ),4)

        metrics["precision_weighted"] = round(precision_score(y_true, y_pred, average='weighted'), 4 )
        metrics["precision_micro"] = round(precision_score(y_true, y_pred, average='micro'), 4 )
        metrics["precision_macro"] = round(precision_score(y_true, y_pred, average='macro'), 4 )
        metrics["recall_weighted"] = round(recall_score(y_true, y_pred, average='weighted'), 4 )
        metrics["recall_micro"] = round(recall_score(y_true, y_pred, average='micro'), 4 )
        metrics["recall_macro"] = round(recall_score(y_true, y_pred, average='macro'), 4 )
        metrics["f1_weighted"] = round(f1_score(y_true, y_pred, average='weighted'), 4 )
        metrics["f1_micro"] = round(f1_score(y_true, y_pred, average='micro'), 4 )
        metrics["f1_macro"] = round(f1_score(y_true, y_pred, average='macro'), 4 )
        metrics["accuracy"] = round(accuracy_score(y_true, y_pred), 4 )
        
        return metrics
