import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV, cross_val_score

class Classifier:
    def __init__(self, categorical_variables, continuous_variables):
        self.categorical_variables = categorical_variables
        self.continuous_variables = continuous_variables

        self.model = Pipeline([
            ('preprocessor', ColumnTransformer([
                ('cat', OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore'), categorical_variables),
                ('cont', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('scaler', StandardScaler())
                ]), continuous_variables)
            ])),
            ('feature_selector', SequentialFeatureSelector(
                LogisticRegression(), 
                n_features_to_select='auto',
                tol=0, 
                scoring='roc_auc')
            ),
            ('classifier', LogisticRegression(penalty='l2'))
        ])
        self.grid_search = GridSearchCV(
            self.model,
            {'classifier__C': np.logspace(-4,4,10)},
            scoring='roc_auc',
            n_jobs=-1
        )

    def fit(self, X, y):
        self.grid_search.fit(X, y)
        self.model.set_params(**self.grid_search.best_params_)
        self.model.fit(X, y)

        self.classes_ = self.model.classes_
        
        selected_features.append(
            self.model['preprocessor'].get_feature_names_out()[
                self.model['feature_selector'].get_support()
            ]
        )

    def predict_proba(self, X):
        preds = self.model.predict_proba(X)
        return preds

    def get_params(self, *args, **kwargs):
        return {
            'categorical_variables':self.categorical_variables,
            'continuous_variables':self.continuous_variables
        }

    def set_params(self, *args, **kwargs):
        return ClinicalModel(self.categorical_variables, self.continuous_variables)