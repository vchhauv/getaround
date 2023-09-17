import pandas as pd
import time
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":

    # Set your variables for your environment
    EXPERIMENT_NAME="GetAroundPricing"

    # Set experiment's info 
    mlflow.set_experiment(EXPERIMENT_NAME)
    # Get our experiment info
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    print("training model...")
    
    # Time execution
    start_time = time.time()

    # Call mlflow autolog
    mlflow.sklearn.autolog() 

    # Import dataset
    df = pd.read_csv("get_around_pricing_project.csv")
    df = df.drop("Unnamed: 0", axis=1)
    # X, y split 
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Train / test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    numeric_features = []
    categorical_features = []

    for i,t in X.dtypes.items():
        if ('float' in str(t)) or ('int' in str(t)) :
            numeric_features.append(i)
        else:
            categorical_features.append(i)

    numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
            ])
    categorical_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore')) 
            ])

    preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ])
        
    model = Pipeline(steps=[("Preprocessing", preprocessor),
                            ("Regressor", LinearRegression())
                            ])

    # Log experiment to MLFlow

    with mlflow.start_run(experiment_id = experiment.experiment_id):
        model.fit(X_train, y_train)
        predictions = model.predict(X_train)

        # Log model seperately to have more flexibility on setup 
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="modeling_car_price",
            registered_model_name="car_price",
            signature=infer_signature(X_train, predictions)
        )
        
    print("...Done!")
    print(f"---Total training time: {time.time()-start_time}")