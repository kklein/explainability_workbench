import pandas as pd
from causalml.inference.meta import BaseRRegressor
from xgboost import XGBRegressor


def get_model():
    df = pd.read_csv("./data/learning_mindset.csv")

    outcome_column = "achievement_score"
    treatment_column = "intervention"
    feature_columns = [
        column
        for column in df.columns
        if column not in [outcome_column, treatment_column]
    ]
    
    r_learner = BaseRRegressor(learner=XGBRegressor(random_state=42))
    r_learner.fit(
        X=df[feature_columns], treatment=df[treatment_column], y=df[outcome_column]
    )
    tau = r_learner.models_tau[1]
    return tau
