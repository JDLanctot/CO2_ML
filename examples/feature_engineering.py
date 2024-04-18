from predictionrl.util import data_to_dictionary, data_to_dataframe, dictionary_metrics
from predictionrl.processing import prep_data
from predictionrl.modeling import model_train_features, feature_importance_from_model, evaluate_model
from predictionrl.clustering import run_clustering

def main():
    data_df = data_to_dataframe('examples\data_python.mat')
    data_dict = data_to_dictionary('examples\data_python.mat')
    metrics_df = dictionary_metrics(data_dict, data_df)
    cluster_df = run_clustering(metrics_df, 'Mean', 'StDev', clusters=4, visualize=True)

    colname = 'Result'
    cols_to_drop = ['MarketId', 'ContractId']
    df = prep_data(metrics_df, colname, cols_to_drop)

    model_str = "forrest"
    if model_str == "lasso":
        model_type = "LassoCV"
    elif model_str == "forrest":
        model_type = "RandomForestRegressor"
    elif model_str == "tree":
        model_type = "DecisionTreeRegressor"
    else:
        model_type = "LinearRegression"

    model, train_set, test_set = model_train_features(df, colname, model_str)

    # SEE HOW THE MODEL RELATES TO THE FEATURES
    feature_names = [col for col in train_set.columns if col != colname]
    importance = feature_importance_from_model(model, feature_names)
    print('-'*90)
    print('Here are our importance results:')
    print(importance)

    # EVALUATE THE MODEL
    evaluate_model(model, train_set, test_set, colname)
    print('-'*90)
    print(f'Here we have used the {model_type} model to make a model which predicts the {colname} based on a number of features within our dataset. Specifically, we have feature engineered descriptive statistics from a early portion of each time series and used those new features within our model to predict the outcome of the contract. Additionally, we have made it possible to run our script via CLI which could be useful in automated pipelines. To improve this, we could accept command line arguments for the filename, column name which should be predicted, and which model should be used. Lastly, I added type safety to all of the function arguments and returns to make sure some potential bugs can be caught through linting.')

if __name__ == "__main__":
    main()
