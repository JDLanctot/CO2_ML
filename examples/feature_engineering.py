from predictionrl.util import data_to_dictionary, data_to_dataframe, dictionary_metrics
from predictionrl.modeling import model_train_features, prep_data, feature_importance_from_model, evaluate_model

def main():
    data_df = data_to_dataframe('examples\data_python.mat')
    data_dict = data_to_dictionary('examples\data_python.mat')
    metrics_df = dictionary_metrics(data_dict, data_df)

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
    print(f'Here we have used the {model_type} model to make a model which predicts the {colname} based on a number of features within our dataset. Specifically, we have feature engineered lattitude and longitude to be combined into a density heatmap weighted by {colname} and used that new feature within our model to improve the results obtained by the example data pipeline. Additionally, we have made it possible to run our script via CLI which could be useful in automated pipelines. To improve this, we could accept command line arguments for the filename, column name which should be predicted, and which model should be used, and instead of showing plots, save them as files so no popup is generated -- thereby pausing the script until a user closes the figure. Lastly, I added type safety to all of the function arguments and returns to make sure some potential bugs can be caught through linting.')


    import ipdb
    ipdb.set_trace()

if __name__ == "__main__":
    main()
