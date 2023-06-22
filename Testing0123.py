def predict_fracture(full_df, parameter):
    #import libraries
    import sys
    sys.path.append('/lakehouse/default/Files')
    sys.path.append('../')

    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import warnings

    """" NOTE: Import packages which customized by VPI (therefore can't be installed with "pip") """
    from Devtools.LightGBM._ligthgbmR import Train_LGBM
    from Devtools.LightGBM.score_cal import RScore
    from Devtools.XGBoost._xgboostR import Train_XGBR
    from Devtools.XGBoost.score_cal import RScore

    pd.set_option('display.max_columns', 100)
    pd.set_option('use_inf_as_na',True)
    warnings.filterwarnings('ignore')

    import joblib
    from datetime import datetime

    seed = 42
    df = full_df
    print(df)
    col = list(df.columns)
    if 'DEPT' in col:
        df['DEPTH']=df['DEPT'].copy()
        df = df.drop(['DEPT'], axis=1)
    target_list = ['NPHI', 'RHOB', 'DTS', 'DTC']
    scoring_list = ['R2', 'MAE', 'MSE', 'RMSE', 'MAPE', 'Poisson', 'Tweedie', 'MeAE', 'ExVS', 'MSLE', 'ME', 'Gamma', 'D2T', 'D2Pi', 'D2A']
    obj_list =['valid_score', 'train_valid_drop']
    algorithm_list = ["lightgbm", "xgboost", "catboost"]
    well_list = list(df['WELL'].unique())
    if len(well_list) !=1:
        well_list.append('all data')
    else:
        well_list = well_list

    # ## 1. Exploratory Data Analysis

    # In[4]:
    well_view = 'all data'
    # ### 1.1. Curve missing percentage
    if well_view=='all data':
        data_view=df
    else:
        data_view = df.loc[df['WELL'].astype(str) == well_view]
        #replace -999 in dataframe
    def replace_999(df,col):
        df[col]=df[col].replace(-999, np.nan)
        return df

    col = [col for col in data_view.columns if col not in ['WELL', 'DEPTH']]
    replace_999(data_view, col)


    # ## 2. Missing log model

    # You can choose wells from list below. In case choosing all well, please set well =['all data']


    well = ['01-97-HXS-1X','15-1-SN-1X', '15-1-SN-2X','15-1-SN-4X', '15-1-SNN-1P', '15-1-SNN-2P','15-1-SNN-3P','15-1-SNN-4P','15-1-SNS-7P','15-1-SNS-4P','15-1-SNS-2P']


    # ### 2.1. Preprocessing

    # If you choose single well, you can choose depth interval for training. Otherwise, please type 'none'.
    data = df
    if len(well)!=1:
        print("Please type 'none' in from_training and to_training")
    else:
        data = data.sort_values(by=['DEPTH'])
        print('Min dept:',data['DEPTH'].min())
        print('Max dept:', data['DEPTH'].max())


    target = 'DTC'
    good_data = 'True'
    upper_interval = '2.5'
    lower_interval = '1.5'
    from_training = 'none'
    to_training = 'none'

    if from_training == 'none':
        data=data
    else:
        data= data.loc[(data['DEPTH'] <= float(to_training))&(data['DEPTH'] >= float(from_training))]
    #replace -999 in dataframe
    def replace_999(df,col):
        df[col]=df[col].replace(-999, np.nan)
        return df
    #replace negative in columns
    def repl_negative(df,col):
        df[col]= np.where(df[col] <0,np.nan, df[col])
        return df

    col = [col for col in data.columns if col not in ['WELL', 'DEPTH']]
    replace_999(data, col)
    if 'BS' in col:
        data['DCALI_FINAL'] = np.where(data['CALI'].isnull(), np.nan, (data['CALI']-data['BS']))
    else:
        data['DCALI_FINAL'] = data['DCALI_FINAL']

    check_negative = ['RHOB', 'LLD', 'LLS', 'DTC', 'DTS']

    repl_negative(data, check_negative)

    if good_data == 'True':
        data = data.loc[(data['DCALI_FINAL'] <= float(upper_interval))&(data['DCALI_FINAL'] >= float(lower_interval))]
    else:
        data=data


    feature_list = [col for col in data.columns if col not in [target, 'WELL']]


    print('Done processing!')
    print('You can choose features in this list:',feature_list)


    # In[25]:


    # feature= ['NPHI', 'RHOB', 'DTS']


    feature = parameter.get("feature")
    # feature = features


    # ### 2.2 Model building

    scoring = parameter.get("scoring")

    objective = parameter.get("objective")

    algorithm = parameter.get("algorithm") #algorithm #'xgboost' #'catboost', #'xgboost' #'lightgbm', 'catboost'

    show_shap = parameter.get("show_shap")

    iteration = parameter.get("iteration")


    # scoring = scoring
    #
    # objective = objective
    #
    # algorithm = algorithm  # algorithm #'xgboost' #'catboost', #'xgboost' #'lightgbm', 'catboost'
    #
    # show_shap = show_shap
    #
    # iteration = iteration
    #save parameters to parameter file
    section2new = {'target': target,'good_data': good_data,'upper_interval': upper_interval, 'lower_interval': lower_interval, 'scoring': scoring, 'objective': objective, 'algorithm': algorithm, 'show_shap': show_shap,'iteration': iteration}
    #check for update
    # if section2!=section2new:
    #     section2.update(section2new)
    #     with open('parameter_section_2.py', 'w') as f:
    #         f.write('section2 = ' + str(section2) + '\n')

    drop = feature.copy()
    drop.append(target)
    drop
    data = data.dropna(how ='any', subset=drop)
    if objective == 'valid_score':
        objective = 0
    else:
        objective = 1
    if target in check_negative:
        if algorithm == 'xgboost':
            task = 'reg:gamma'
        elif algorithm == 'lightgbm':
            task = 'gamma'
        else:
            task ='MAE'
    else:
        if algorithm == 'xgboost':
            task = 'reg:squarederror'
        elif algorithm == 'lightgbm':
            task = 'regression'
        else:
            task = 'RMSE'
    print(data)
    y=data[target]
    X=data[feature]
    #print(X.isna().sum())
    #print(feature)
    #split data into sets
    X_use, X_test, y_use, y_test = train_test_split(X, y, train_size=0.9, random_state=seed, shuffle=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X_use, y_use, train_size=0.8, random_state=seed, shuffle=True)
    preprocessors = Pipeline(steps=
            [
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("scaling", MinMaxScaler())
        ]
    )
    #print(X_train.shape)
    X_train, X_valid = preprocessors.fit_transform(X_train), preprocessors.transform(X_valid)
    X_test = preprocessors.transform(X_test)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    #print(X_train.shape)
    #print(X.shape)
    #print(feature)
    X_train = pd.DataFrame(X_train, columns=feature)
    X_valid = pd.DataFrame(X_valid, columns=feature)
    X_test  = pd.DataFrame(X_test, columns=feature)

    if algorithm=='lightgbm':
        model = Train_LGBM(
            features = X_train,
            target = y_train,
            iterations = int(iteration),
            scoring = scoring,
            validation_size = 0.1,
            task = task,
            )
        y_pred = model.predict(X_test)
        score = RScore(y_true=y_test, y_pred=y_pred, scoring=scoring)
        print(score)
        # model_best = model["lightgbm"]
    elif algorithm =='xgboost':
        model = Train_XGBR(
            features = X_train,
            target = y_train,
            iterations = int(iteration),
            scoring = scoring,
            validation_size = 0.1,
            # base_score=0.5,
            # test_set= (X_test, y_test),
            task = task,
            # objectives = objective,
            # show_shap = show_shap,
            # refit = False,
            # saved_dir='xgboost_shaps'
            )
        # model_best = model["xgboost"]
        y_pred=model.predict(xgb.DMatrix(data=X_test, label=y_test))
        score = RScore(y_true=y_test, y_pred=y_pred, scoring=scoring)
        score_Train = RScore(y_true=y_test, y_pred=y_pred, scoring=scoring)
        print(f"Test score: {score}")
    else:
        model= Train_CATR(
            features=X_train,
            target=y_train,
            iterations = int(iteration),
            base_score=0.5,
            scoring=scoring,
            validation_size = 0.1,
            # test_set=(X_test, y_test),
            task=task,
            # objectives=objective, #{0: "valid_score", 1: "train_valid_drop"}
            # show_shap=show_shap, # flag to show shap True or False
            # refit=False,
            )
        y_pred=model.predict(cat.Pool(data=X_test, label=y_test))
        score = RScore(y_true=y_test, y_pred=y_pred, scoring=scoring)
        score_Train = RScore(y_true=y_test, y_pred=y_pred, scoring=scoring)
        print(f"Test score: {score}")

    if algorithm =='xgboost':
        model.save_model('/lakehouse/default/Files/Saved_Models/model_json_Goal2.json')
        import json
        with open('/lakehouse/default/Files/Saved_Models/model_json_Goal2.json') as f:
            data = json.load(f)
        print(data)
        # Convert the model output to a JSON string
        model_json_str = json.dumps(data)

    elif algorithm=='lightgbm':
        import json
        model_json = model.dump_model()
        # Convert the model output to a JSON string
        model_json_str = json.dumps(model_json)
    #
    else:
        model.save_model('/lakehouse/default/Files/Saved_Models/2model_json.json')
        import json
        with open('/lakehouse/default/Files/Saved_Models/2model_json.json') as f:
            data = json.load(f)
        print(data)
        #Convert the model output to a JSON string
        model_json_str = json.dumps(data)

    if algorithm=='lightgbm':
        full_predict = model.predict(data=X)
        arr = np.array(full_predict)
        json_str = json.dumps(arr.tolist())
    else:
        full_predict = model.predict(xgb.DMatrix(data=X, label=y))
        dataset_full = full_predict

        arr = np.array(dataset_full)
        json_str = json.dumps(arr.tolist())
    # result_in_json_1 = json_str.to_json(orient='index')
#API return for output
    custom_result = {}

    if parameter.get("Testing_score"):
        custom_result["Testing_score"] = score

    if parameter.get("Train_score"):
        custom_result["Train_score"] = score_Train

    if parameter.get("Modeling_result"):
        custom_result["Modeling_result"] = model_json_str

    if parameter.get("Predicted_Results"):
        custom_result["Predicted_Results"] = json_str

    result_in_json_full = {
        # "Modeling_result": model_json_str,
        # "Predicted_Results": json_str,
        **parameter,
        **custom_result
    }
    import json
    result_in_json = json.dumps(result_in_json_full)
    return result_in_json
