import os
import pandas as pd
import numpy as np

################################################################################
# Read well logs from ASC data files and merge them
################################################################################

def read_asc(file):
    lines = open(file).readlines()
    columns = [x.strip() for x in lines[1].split()]

    rows = []
    for line in lines[4:]:
        row =[x.strip() for x in line.split()]
        if len(row) == len(columns):
            rows.append(row)
    df = pd.DataFrame(rows, columns=columns).apply(pd.to_numeric)

    # Replace negative values by NaN
    df.mask(df < 0, np.NaN, inplace=True)

    # Use UPPER CASE column names
    df.columns = [col.upper() for col in df.columns]

    # Interpreted logs
    df.rename(columns={'VWCL': 'VCL_INT', 'PHIE': 'PHI_INT', 'SW': 'SW_INT'}, inplace=True)
    df.rename(columns={'VCL_CORE': 'VCL', 'PHI_CORE': 'PHI', 'PERM_CORE': 'PERM', 'SW_CORE': 'SW'}, inplace=True)

    return df

def read_data(data_path):
    data_path = '/lakehouse/default/Files'
    # files = [(i, os.path.join(data_path, f'{i}.asc')) for i in range(1, 100)]
    files = [(i, os.path.join(data_path, '{}.asc'.format(i))) for i in range(1, 100)]
    files = [x for x in files if os.path.isfile(x[1])]
    # print(f'Total {len(files)} data files')
    print('Total {} data files'.format(len(files)))

    dfs = []
    for well_no, file in files:
        print(file)
        df = read_asc(file)
        cols = list(df.columns)
        df['WELL_NO'] = well_no
        df = df[['WELL_NO'] + cols]
        # print(f'{len(df)} rows\n')
        print('{} rows\n'.format(len(df)))
        dfs.append(df)

    # Merge dataframes
    df = pd.concat(dfs)
    df.reset_index(drop=True, inplace=True)
    return df

################################################################################
# Extract data for each of dependent variable and save to a HDF5 file
################################################################################

# Independent Variables
IV_COLS = ['DEPTH', 'GR', 'LLD', 'LLS', 'RHOB', 'NPHI', 'DT']

# Dependent Variables
DV_COLS = ['VCL', 'PHI', 'PERM', 'SW']

def extract_data(big_df, gen_report=False):
    for dv_col in DV_COLS:
        cols = ['WELL_NO'] + IV_COLS + [dv_col]
        int_col = dv_col + '_INT' # Interpreted log
        if int_col in list(big_df.columns):
            cols += [int_col]

        df = big_df.copy()[cols]

        ### Well no.12 has invalid values in PHI & SW columns!!!
        if dv_col in ['PHI', 'SW']:
            df = df[df['WELL_NO'] != 12]

        df.dropna(subset=[dv_col], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # print(f'\n*** {dv_col} ***')
        # print(list(df.columns))
        # print(f'Total {len(df)} rows')
        # print(df.groupby(['WELL_NO'])['WELL_NO'].count())
        print('\n*** {} ***'.format(dv_col))
        print(list(df.columns))
        print('Total {} rows'.format(len(df)))
        print(df.groupby(['WELL_NO'])['WELL_NO'].count())

        # Check for invalid values
        out_wells = list(df[(df[dv_col] < 0) | (df[dv_col] > 1)]['WELL_NO'].unique())
        if out_wells:
            print('Out of range [0..1] wells:', out_wells)

        # Save dataframe to a HDF5 file
        os.makedirs('data_h5', exist_ok=True)
        # df.to_hdf(f'data_h5\\{dv_col}.h5', key=dv_col)
        df.to_hdf('data_h5/{}.h5'.format(dv_col), key=dv_col.replace('/', '_'))

        # Generate ProfileReport
        if gen_report:
            from pandas_profiling import ProfileReport
            os.makedirs('reports', exist_ok=True)
            ProfileReport(df).to_file(output_file='reports/{}.html'.format(dv_col))
            print()

################################################################################

def main():
    df = read_data('data')
    print('*** Merged dataframe ***')
    print(list(df.columns))
    print('Total {} rows'.format(len(df)))
    extract_data(df, False)

if __name__ == '__main__':
    main()
