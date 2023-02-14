import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV
from sksurv.linear_model import CoxPHSurvivalAnalysis
from util import concordance_index


class BaseModel:
    def __init__(self, categorical_variables, continuous_variables):
        self.categorical_variables = categorical_variables
        self.continuous_variables = continuous_variables

        self.preprocessor = (
            'preprocessor', 
            ColumnTransformer([
                ('cat', OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore'), categorical_variables),
                ('cont', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('scaler', StandardScaler())
                ]), continuous_variables)
            ]))

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_

    def predict_proba(self, X):
        preds = self.model.predict_proba(X)
        return preds

    def get_params(self, *args, **kwargs):
        return {
            'categorical_variables':self.categorical_variables,
            'continuous_variables':self.continuous_variables
        }

    def set_params(self, *args, **kwargs):
        return Classifier(self.categorical_variables, self.continuous_variables)


class Classifier(BaseModel):
    def __init__(self, categorical_variables, continuous_variables):
        super().__init__(categorical_variables, continuous_variables)

        self.pipeline = Pipeline([
            self.preprocessor,
            (
                'feature_selector', 
                SequentialFeatureSelector(
                    LogisticRegression(), 
                    n_features_to_select='auto',
                    tol=0, 
                    scoring='roc_auc'
                )
            ),
            ('classifier', LogisticRegression(penalty='l2'))
        ])
        self.model = GridSearchCV(
            self.pipeline,
            {'classifier__C': np.logspace(-4,4,10)},
            scoring='roc_auc',
            n_jobs=-1
        )

    def set_params(self, *args, **kwargs):
        return Classifier(self.categorical_variables, self.continuous_variables)


class SurvivalModel(BaseModel):
    def __init__(self, categorical_variables, continuous_variables):
        super().__init__(categorical_variables, continuous_variables)

        self.model = Pipeline([
            self.preprocessor,
            (
                'feature_selector', 
                SequentialFeatureSelector(
                    CoxPHSurvivalAnalysis(), 
                    n_features_to_select='auto',
                    tol=0, 
                    scoring=concordance_index
                )
            ),
            ('classifier', CoxPHSurvivalAnalysis())
        ])