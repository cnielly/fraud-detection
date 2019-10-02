from sklearn.preprocessing import RobustScaler
import pandas as pd



def scaling(df, feats, scaler):
    """
    Function to scale set of features using a scaler. Create new feature named 'scaled_' + Feature name

    :input df: pd.DataFrame to which apply scaling on
    :input feats: list of variables to apply sclaing on
    :input scaler: SK Learn scaling 

    :output df: df with new feature column and without initial features

    """
    scaler = scaler
    #Scale selected features
    if set(feats).issubset(df.columns.to_list()):
        for feat in feats:
            df['scaled_' + feat] = scaler.fit_transform(df[feat].values.reshape(-1,1))
        #Drop features
        df.drop(feats, axis=1, inplace=True)
        return df
    else: 
        print('Column name not recognised')
        return None
    return df


def get_xy(df, target):
    """
    Function to get X and y data
    :input df: pd.DataFrame to extract X and y data
    :input target: string of target name to create y 
    """
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y
    