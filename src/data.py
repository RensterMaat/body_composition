import pandas as pd
from pathlib import Path

BODY_COMPOSITION_PREDICTORS = ['BMI','SMI','SMD','SMG','TATI','SATI','VATI']
CLINICAL_PREDICTORS = ['ajcc_stage','ldh','primary_location','tfdr','satellites','who']

CATEGORICAL_VARIABLES = [p for p in CLINICAL_PREDICTORS if not p =='tfdr']
CONTINUOUS_VARIABLES = BODY_COMPOSITION_PREDICTORS + ['tfdr']

DATA_CSV_PATH = Path('/home/rens/repos/body_composition/data/data.csv')


if __name__ == '__main__':
    data = pd.read_csv('/mnt/d/quantib/measurements/L3_features_all_with_names.csv', sep=';')
    data = data[[
        'Patient name',
        'Height',
        'Weight',
        'Subcutaneous fat',
        'Visceral fat',
        'Psoas muscle',
        'Abdominal muscle',
        'Long spine muscle',
        'Mean HU Psoas muscle',
        'Mean HU Abdominal muscle',
        'Mean HU Long spine muscle'
    ]].set_index('Patient name')

    data.index = [ix.replace('-','_') for ix in data.index]

    data['Height'] = data['Height'] / 100

    # body mass index
    data['BMI'] = data['Weight'] / data['Height'] ** 2

    # skeletal muscle index
    data['SMI'] = (data['Psoas muscle'] + data['Abdominal muscle'] + data['Long spine muscle']) / data['Height']

    # skeletal muscle density
    muscle_areas = data[['Psoas muscle','Abdominal muscle','Long spine muscle']]
    muscle_densities = data[['Mean HU Psoas muscle','Mean HU Abdominal muscle','Mean HU Long spine muscle']]

    data['SMD'] = (muscle_areas.to_numpy() * muscle_densities.to_numpy()).sum(axis=1) / muscle_areas.sum(axis=1)

    # skeletal muscle gauge
    data['SMG'] = data['SMI'] * data['SMD']

    # total adipose tissue index
    data['TATI'] = (data['Subcutaneous fat'] + data['Visceral fat']) / data['Height']

    # subcutaneous adipose tissue index
    data['SATI'] = data['Subcutaneous fat'] / data['Height']

    # visceral adipose tissue index
    data['VATI'] = data['Visceral fat'] / data['Height']

    dmtr = pd.read_csv('/mnt/c/Users/user/data/tables/dmtr.csv').set_index('id')
    data = data[BODY_COMPOSITION_PREDICTORS].join(dmtr[['dcb','response','fu_OS','event_OS','fu_PFS','event_PFS']])

    data['fu_OS'] = data['fu_OS'].apply(lambda x: int(x.split()[0]) if type(x) == str else float('nan'))
    data['fu_PFS'] = data['fu_PFS'].apply(lambda x: int(x.split()[0]) if type(x) == str else float('nan'))

    stage = []
    for ix, row in dmtr.iterrows():
        if row.Stage == 'M1d':
            if row.hersenmet == 1:
                stage.append('M1d_symptomatic')
            else:
                stage.append('M1d_asymptomatic')
        elif pd.isna(row.Stage):
            stage.append('missing')
        else:
            stage.append(row.Stage)

    dmtr['ajcc_stage'] = stage

    dmtr["ldh"] = (
        dmtr.labldh + (dmtr.labldhw > 500).astype(int)
    ).replace(
        {0: "missing", 1: "normal", 2: "elevated", 3: ">2x ULN", 9: "missing"}
    ).fillna('missing')

    dmtr['primary_location'] = dmtr.ptloc.replace({
        0:'unknown',
        1:'ocular',
        2:'head-neck',
        3:'trunk',
        4:'extremity',
        5:'acral',
        6:'mucosoal',
        9:'missing'
    }).fillna('missing')

    dmtr['tfdr'] = (pd.to_datetime(dmtr.dathure) - pd.to_datetime(dmtr.datprim)).dt.days
    dmtr.loc[dmtr[dmtr.primary_location.isin(['unknown','missing'])].index, 'tfdr'] = float('nan')

    dmtr['satellites'] = dmtr.sattel.replace({
        0:'0',
        1:'1',
        2:'1',
        3:'1',
        9:'missing'
    }).fillna('missing')


    dmtr["who"] = dmtr.WHO.replace([2, 3, 4], "2-4").replace(
        {0: "0", 1: "1", float('nan'):'missing'}
    )

    data = data.join(dmtr[CLINICAL_PREDICTORS + ['center']])

    data.to_csv('/home/rens/repos/body_composition/data/data.csv')