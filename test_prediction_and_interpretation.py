import os
import pandas as pd
import plotly
import plotly.graph_objects as go
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

IV_COLS = ['DEPTH', 'GR', 'LLD', 'LLS', 'RHOB', 'NPHI', 'DT']
DV_COLS = ['VCL', 'PHI', 'PERM', 'SW']

################################################################################
# Regressors
################################################################################

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lars
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import OrthogonalMatchingPursuit

regressors = {
    "XGBRegressor": XGBRegressor(),
    "RandomForestRegressor": RandomForestRegressor(),
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "SVR": SVR(),
    "NuSVR": NuSVR(),
    "Ridge":Ridge(),
    "Lars": Lars(normalize=False),
    "HuberRegressor": HuberRegressor(max_iter=500),
    "ARDRegression": ARDRegression(),
    "BayesianRidge": BayesianRidge(),
    "ElasticNet": ElasticNet(),
    "OrthogonalMatchingPursuit": OrthogonalMatchingPursuit(normalize=False),
}

################################################################################
# Plot line chart
################################################################################

def plot_chart(dv_col, name, result):
    plot_name = f'{dv_col}_{name}'
    plot_title = f'{dv_col}: {len(result)} rows, {name}'

    layout = go.Layout(title=plot_title, yaxis=dict(title=dv_col), showlegend=True)
    fig = go.Figure(layout=layout)

    fig.add_trace(go.Scatter(x=result.index, y=result['CORELOG'], name='CORE LOG', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=result.index, y=result['PREDICTION'], name='PREDICTION', mode='lines+markers'))
    if 'INTERPRETATION' in result.columns:
        fig.add_trace(go.Scatter(x=result.index, y=result['INTERPRETATION'], name='INTERPRETATION', mode='lines+markers'))

    os.makedirs('charts', exist_ok=True)
    plotly.offline.plot(fig, filename=f'charts\\{plot_name}.html',  auto_open=False, config={'scrollZoom': True})

################################################################################
# Scoring models
################################################################################

N_SPLITS = 5
RANDOM_STATE = 12345
SHUFFLE = True

def score_model_kfold(model, df, IV_COLS, dv_col, int_col):
    name, reg_model = model
    kf = KFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=SHUFFLE)
    X = df[IV_COLS]
    y = df[dv_col]

    all_parts = []
    for train_index, test_index in kf.split(X):
        X_train , X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        reg_model.fit(X_train, y_train)
        y_predict = reg_model.predict(X_test)
        arr = np.array([y_test, y_predict])
        part_df = pd.DataFrame(data = arr.transpose(), index = test_index, columns = ['CORELOG', 'PREDICTION'])
        all_parts.append(part_df)

    result_df = pd.concat(all_parts).sort_index(ascending=True)
    result_df.loc[result_df['PREDICTION'] < 0, 'PREDICTION'] = 0
    if int_col in list(df.columns):
        result_df['INTERPRETATION'] = df[int_col]
    return (name, result_df)

def calc_rmse(target, predict):
    return np.sqrt(mean_squared_error(target, predict))

def test_col(df, dv_col, int_col):
    models =list(regressors.items())
    for model in models:
        name, result = score_model_kfold(model, df, IV_COLS, dv_col, int_col)
        rmse = calc_rmse(result['CORELOG'], result['PREDICTION'])
        r2 = r2_score(result['CORELOG'], result['PREDICTION'])
        print(f'{name:25} rmse = {rmse:<10.02f} r2 = {r2:.02f}')

        if name in ['XGBRegressor', 'RandomForestRegressor']:
            plot_chart(dv_col, name, result)

    if int_col in list(df.columns):
        rmse = calc_rmse(df[dv_col], df[int_col])
        r2 = r2_score(df[dv_col], df[int_col])
        print(f'{">> Interpretation":25} rmse = {rmse:<10.02f} r2 = {r2:.02f}')
    else:
        print('(no interpretation)')

def test(dv_col):
    df_col = pd.read_hdf(f'data_h5\\{dv_col}.h5', key=dv_col)

    int_col = dv_col + '_INT' # Interpretation
    if int_col in list(df_col.columns):
        df_col = df_col[['WELL_NO'] + IV_COLS + [dv_col, int_col]]
    else:
        df_col = df_col[['WELL_NO'] + IV_COLS + [dv_col]]

    df_col.dropna(inplace=True)
    df_col.reset_index(drop=True, inplace=True)

    print(len(df_col), 'rows')
    test_col(df_col, dv_col, int_col)

################################################################################

def main():
    for dv_col in DV_COLS:
        print(f'\n*** {dv_col} ***')
        test(dv_col)

if __name__ == '__main__':
    main()
