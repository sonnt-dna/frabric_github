import pandas as pd
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
# Grouping
# - Case 1: by well number
# - Case 2: by well number and depth
################################################################################

well_groups_1 = {
    1: [1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37],
    2: [8],
    3: [9, 10, 11],
    4: [12, 13, 14, 15, 38],
    5: [16, 17],
    6: [18, 19],
    7: [20],
    8: [21, 22, 23, 24, 25, 26, 27, 28],
    9: [29, 30, 31],
}

well_groups_2 = {
    1: [10, 13, 14, 15, 29, 30, 31, 20, 23, 24, 25, 26, 27, 28, 16, 17, 18],
    2: [8, 12, 13, 15, 20, 21, 22, 23, 16, 17, 18],
    3: [8, 10, 11, 13, 15, 29, 21, 23, 16, 18],
    4: [1, 2, 3, 4, 6, 9, 10, 11, 13, 21, 22, 23, 24, 27, 28, 16, 17, 18],
    5: [1, 2, 3, 5, 7, 10],
}

dtable = """
Group 	1	2	3	4	5	6	7	8	9	10	11	12	13	14	15	16	17	18	19	20	21	22	23	24	25	26	27	28	29	30	31	32	33	34	35	36	37	38
1	1976.4	1990.1	1983	1967.5	2001	2049.4	2195	1830.9	1671.5	1598		1718.8	1701.5	1812	1684	1921.4	2102.7	2130	2190	2113.47	1977.28	2070.12	2103.05	2108.91	2107.35	2184.71	2684.15	2554.35	1666.6	1811.39	1811.39	1991	2096.2	2049.4	2006.2	2258	2027	1697
2	2638.4	2891.6	2616	2600	2634.3	2741.4	2870.7	2320.65	2067	1963		2201.2	2148.5	2394.5	2112.4	2006.6	2205.9	2337.25	2397.72	2368.21	2209.65	2282.11	2452.22	2351.94	2337.6	2501.4	3084.78	2814				2622	2857	2741.4	2627	3163	2726
3	2981	3229.4	2850.8	2850.3	2891.3	3085.8	3134	2577.15	2110	1985		2373	2282	2584.1	2211	2131.7	2358.6	2490	2593.25	2482.38	2409.14	2479.32	2532.14	2526.03	2541	2607.68	3236.08	2941.5	2438.66	2655.1	2655.1	2887.5	3205	3085.8	2837		2996
4	3718	3449.6	3677.6	3840.3	3780	4011.6	3960.3						2659			3254	3295.3	3297.42	3113.87		2659.83	2761.45	2949.43	2852.47	2783.1	2860.14	3536.3	3274.65				3781.8	4227.9	4011.6	3637	4407	4136.7
5	3969	3723.6	4074	4393.9	3959.9	4339.9	4392.2																									4048.8	4347.5	4339.9	3879	4576	4214.2
"""

lines = [[y.strip() for y in x.split('\t')] for x in dtable.split('\n') if x]
total_wells = len(lines[0]) - 1
total_groups = len(lines) - 1
assert lines[0][0] == 'Group' and lines[0][-1] == str(total_wells)

def to_float(x):
    try:
        return float(x)
    except:
        return -np.inf

wgdict = {}

def wgadd(group, well, value):
    gdict = wgdict.get(well, {0:0.0})
    gdict[group] = value
    wgdict[well] = gdict

for group in range(1, total_groups + 1):
    line = [to_float(x) for x in lines[group]]
    max_well = len(line) - 1
    assert max_well <= total_wells
    for well in range(1, max_well + 1):
        value = line[well]
        if value != -np.inf:
            wgadd(group, well, value)

def group_from_well_and_depth(row):
    well = row['WELL_NO']
    depth = row['DEPTH']
    if well not in wgdict:
        return 0
    gdict = wgdict[well]
    groups = sorted(gdict.keys())
    for i in range(len(groups) - 1):
        if gdict[groups[i]] <= depth < gdict[groups[i+1]]:
            return groups[i]
    else:
        return groups[-1]

################################################################################
# Scoring models
################################################################################

N_SPLITS = 5
RANDOM_STATE = 12345
SHUFFLE = True

def score_model_kfold(model, df, IV_COLS, dv_col):
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
    return (name, result_df)

def calc_rmse(target, predict):
    return np.sqrt(mean_squared_error(target, predict))

def test_col(df, dv_col):
    models =list(regressors.items())[:2]
    for model in models:
        name, result = score_model_kfold(model, df, IV_COLS, dv_col)
        rmse = calc_rmse(result['CORELOG'], result['PREDICTION'])
        r2 = r2_score(result['CORELOG'], result['PREDICTION'])
        print(f'{name:25} rmse = {rmse:<10.02f} r2 = {r2:.02f}')

def load_df(dv_col):
    df_col = pd.read_hdf(f'data_h5\\{dv_col}.h5', key=dv_col)

    df_col = df_col[['WELL_NO'] + IV_COLS + [dv_col]]
    df_col.dropna(inplace=True)
    df_col.reset_index(drop=True, inplace=True)

    # Add column for grouping by well number and depth
    df_col['GROUP'] = df_col.apply(lambda x: group_from_well_and_depth(x), axis=1)

    return df_col

def test_groups(case, df_col, dv_col):
    print(f'\n[ All groups ]')
    print(len(df_col), 'rows')
    test_col(df_col, dv_col)

    if case == 1:
        groups = well_groups_1.keys()
    elif case == 2:
        groups = well_groups_2.keys()
    else:
        return

    for group in groups:
        if case == 1:
            df = df_col[df_col['WELL_NO'].isin(well_groups_1[group])]
        else:
            df = df_col[(df_col['GROUP'] == group) & df_col['WELL_NO'].isin(well_groups_2[group])]
        df.reset_index(drop=True, inplace=True)
        print(f'\n[ Group {group} ]')
        print(len(df), 'rows')
        if len(df) >= N_SPLITS*2:
            test_col(df, dv_col)
        else:
            print('Not enough number of rows!')

################################################################################

def main():
    print()
    print('**********************************')
    print('*  Grouping by well number (#1)  *')
    print('**********************************')
    for dv_col in DV_COLS:
        print(f'\n*** {dv_col} ***')
        df_col = load_df(dv_col)
        test_groups(1, df_col, dv_col)

    print()
    print('********************************************')
    print('*  Grouping by well number and depth (#2)  *')
    print('********************************************')
    for dv_col in DV_COLS:
        print(f'\n*** {dv_col} ***')
        df_col = load_df(dv_col)
        test_groups(2, df_col, dv_col)

if __name__ == '__main__':
    main()
