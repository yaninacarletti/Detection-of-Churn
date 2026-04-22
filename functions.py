import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score                       
import warnings
warnings.filterwarnings('ignore')



def data_quality_analysis(data):
    tipos = pd.DataFrame({'tipo': data.dtypes},index=data.columns)
    na = pd.DataFrame({'nulos': data.isna().sum()}, index=data.columns)
    na_prop = pd.DataFrame({'porc_nulos':data.isna().sum()/data.shape[0]},
    		  index=data.columns)
    ceros = pd.DataFrame({'ceros':[data.loc[data[col]==0,col].shape[0] for col in data.columns]},
    		index= data.columns)
    ceros_prop = pd.DataFrame({'porc_ceros':[data.loc[data[col]==0,col].shape[0]/data.shape[0] for col in data.columns]},
    			index= data.columns)
    summary = data.describe(include='all').T

    summary['dist_IQR'] = summary['75%'] - summary['25%']
    summary['limit_inf'] = summary['25%'] - summary['dist_IQR']*1.5
    summary['limit_sup'] = summary['75%'] + summary['dist_IQR']*1.5

    summary['outliers'] = data.apply(lambda x: sum(np.where((x<summary['limit_inf'][x.name]) | (x>summary['limit_sup'][x.name]),1 ,0)) if x.name in summary['limit_inf'].dropna().index else 0)


    return pd.concat([tipos, na, na_prop, ceros, ceros_prop, summary], axis=1).sort_values('tipo')





def plot_descriptive(calidad, df, columns_review = None):
    plt.rcParams.update({'font.size': 8})
    if columns_review:
        columns_distributions = columns_review
    else:
        columns_distributions = df.columns

    plt.figure(figsize=(10, 8))
    number_rows = len(columns_distributions)//2 + len(columns_distributions)%2
    for n, i in enumerate(columns_distributions):
        plt.subplot(number_rows, 2, n + 1)
    if calidad.loc[i, 'tipo']=='object':
        col = df[i].astype(str)
        sns.countplot(y= col, order=col.value_counts().iloc[:7].index)
        plt.title('Frecuencias para {}'.format(i))
        plt.tight_layout()
    else:
        if df[i].dtype == 'float':
            sns.distplot(df[i])
            plt.axvline(np.mean(df[i]), color='tomato')
            plt.title('Distribución para {}'.format(i))
            plt.tight_layout()
        else:
            sns.distplot(df[i])
            plt.title('Distribución para {}'.format(i))
            plt.tight_layout()





def outlier_detection(df, features):
    Q1 = df[features].quantile(0.25)
    Q3 = df[features].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    mask = ((df[features] >= lower) & (df[features] <= upper)).all(axis=1)

    df_clean = df[mask]
    return df_clean





def evaluate_model(model, X_test, y_test): 
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
     
    precision = round(precision_score(y_test, y_pred), 2)   
    recall = round(recall_score(y_test, y_pred), 2)
    f1 = round(f1_score(y_test, y_pred), 2)
    roc_auc = round(roc_auc_score(y_test, y_proba), 2)
    prc_auc = round(average_precision_score(y_test, y_proba), 2)
 
    return (precision, recall, f1, roc_auc, prc_auc, y_proba)


