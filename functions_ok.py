import pandas as pd
pd.set_option("display.max_columns",100)
import warnings
warnings.filterwarnings("ignore")
import re
import unicodedata

from plotly.offline import plot,iplot
pd.options.plotting.backend = "plotly"
import plotly.graph_objects as go
import plotly.express as px#graficos express


import cufflinks as cf
cf.go_offline()
import numpy as np
import nltk

import matplotlib.pyplot as plt
import seaborn as sns


from pyecharts.charts import Pie
from pyecharts import options as opts

from sklearn.model_selection import train_test_split

from matplotlib import pyplot

from statsmodels.stats.outliers_influence import variance_inflation_factor


def clean_text(text, pattern="[^a-zA-Z0-9 ]"):
    """Limpia el texto para facilitar su análisis, remueve acentos, carácteres especiales y espacios innecesarios. También pasa toda la cadena a minúsculas.

    Parameters
    ----------
    text : string
        String que contiene el texto
    pattern : str, optional
        Expresión regular para mantener en el texto, por default es ``[^a-zA-Z0-9 ]``

    Returns
    -------
    cleaned_text : string
        Texto limpio

    Example
    -------
    >>> clean_text('¡Feliz año nuevo, México!')
    >>> u'feliz ano nuevo mexico'
    """
    cleaned_text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
    cleaned_text = re.sub(pattern, " ", cleaned_text.decode("utf-8"), flags=re.UNICODE)
    cleaned_text = u' '.join(cleaned_text.lower().strip().split())
    return cleaned_text



def rename_cols(df,cols,prefix):
    """Renombra columnas de un data frame, con prefijos asignados por el usuario

    Parameters
    ----------
    df : pandas data frame
        Tabla cuyos nombres se quieren cambiar
    cols : list
        lista de nombres de columnas que se van a modificar
    prefix : text
        Prefijo que se desa colocar al listado de variables dado

    Returns
    -------
    df : pandas data frame
        Data frame con los nuevos nombres
    """
    new_feats=[prefix+col for col in cols]
    df=df.rename(columns=dict(zip(cols,new_feats)))
    return df

def calc_vif(df):
    """Dado un data frame con columnas numéricas, calcula el factor de inflación para cada columna 

    Parameters
    ----------
    df : pandas data frame
        Tabla con variables de tipo numérico, a las que se quiere calcular su VIF

    Returns
    -------
    df : pandas data frame
        Data frame que contiene un resumen con las variables y su VIF calculado, además los resultados se entregan en orden descendente por el VIF
    """
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return(vif).sort_values(by="VIF",ascending=False)

def completitud(df):
    """Esta función permite calcular la completitud de las columnas de una tabla

    Parameters
    ----------
    df : pandas data frame
        Tabla a la que se le quieren calcular las completutudes por columna

    Returns
    -------
    df: pandas data frame
        Tabla con el resumen por columna, total de nulos y completitud ordenada de forma ascendente por completitud
    """
    comp=pd.DataFrame(df.isnull().sum())
    comp.reset_index(inplace=True)
    comp=comp.rename(columns={"index":"columna",0:"total"})
    comp["completitud"]=(1-comp["total"]/df.shape[0])*100
    comp=comp.sort_values(by="completitud",ascending=True)
    comp.reset_index(drop=True,inplace=True)
    return comp

def outliers_tests(df,col):
    """Función que busca outliers de una columna de un dataframe por 3 métodos: rango intercuartil, percentiles .05 y .95 y z-score.
    Para este último método aplica una prueba de normalidad, si es rechazada aplica a los datos una tranformación de box-cox para intentar
     volverlos normales y les vuelve a aplicar la prueba de normalidad, si es rechazada, este método se descarta.
     Adicionalmente, grafica la distribución antes y después de la remoción de outliers.

    Parameters
    ----------
    df : pandas data frame
        Tabla que contiene a la columna de interés
    col : numeric column from pandas data frame
        Columna de valores numéricos en la que se quiere detectar la presencia de outliers.

    Returns
    -------
    outliers_index : list
        Lista con los índices de los renglones que fueron catalogados como outliers
    """
    from scipy.stats import shapiro
    from scipy import stats 

    #IQR
    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)
    IQR=Q3-Q1
    IQR_INF=Q1-1.5*(IQR)
    IQR_SUP=Q3+1.5*(IQR)
    IQR_NUM=df[(df[col] < IQR_INF) | (df[col] > IQR_SUP)].shape[0]
    IQR_INDEX=list(df[(df[col] < IQR_INF) | (df[col] > IQR_SUP)].index)

    #Percentiles
    PER_INF=np.percentile(df[col].dropna(),5)
    PER_SUP=np.percentile(df[col].dropna(),95)
    PER_NUM=df[(df[col] < PER_INF) | (df[col] > PER_SUP)].shape[0]
    PER_INDEX=list(df[(df[col] < PER_INF) | (df[col] > PER_SUP)].index)

    print("Para la columna " + col + " se tiene:")
    print("Número de outliers por IQR: " + str(IQR_NUM))
    print("Número de outliers por percentiles .5 y .95: " + str(PER_NUM))

    #Z score
    stat, p = shapiro(df[col])
    if p > 0.05:
        normality = True
        print("Los datos distribuyen normal")
    else:
        normality = False
        print("Los datos no distribuyen normal, se hará una transformación de boxcox")

    if not normality:
        if sum(df[col] > 0) / len(df[col]) == 1:
            fitted_data, fitted_lambda = stats.boxcox(df[col])  
            stat, p = shapiro(fitted_data)
            if p > 0.05:
                normality = True
                print("Los datos transformados distribuyen normal")
            else:
                normality = False

    if normality:
        z=np.abs(stats.zscore(df[col],nan_policy='omit'))
        Z_NUM = df[[col]][(z>=3)].shape[0]
        Z_INDEX=list(df[[col]][(z>=3)].index)
        print("Número de outliers por z-score: " + str(Z_NUM))
    else: 
        Z_NUM = 0
        Z_INDEX = list(df[[col]].index)
        print("los datos no se pudieron tranformar a una normal, no se puede aplicar z-score")

    a=set(IQR_INDEX)
    b=set(PER_INDEX)
    c=set(Z_INDEX)    
    a_=a.intersection(b).intersection(c)
    outliers_index=list(a_)

    print("Hay " + str(len(outliers_index)) + " outliers comúnes = " + str(len(outliers_index)/df.shape[0])+"%")
    
    
    display(df[df.index.isin(outliers_index)][col].describe(percentiles=np.arange(0.1,1.1,.1)))
    
    plt.subplots(figsize=(3,2), dpi=100)
    sns.distplot( df[col] , color="dodgerblue")
    plt.title('Distribución con outliers', backgroundcolor='#565656', fontsize=10, weight='bold',color='white',
                 style='italic',loc='center',pad=30)
    plt.show() 
    
    
    plt.subplots(figsize=(3,2), dpi=100)
    sns.distplot( df[~df.index.isin(outliers_index)][col] , color="green")
    plt.title('Distribución sin outliers', backgroundcolor='#565656', fontsize=10, weight='bold',color='white',
                 style='italic',loc='center',pad=30)
    plt.show() 
    
          
    return outliers_index



from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

from sklearn import metrics

def metrics_clasif_calc(y_test, y_pred, y_score):
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    plt.title('Matriz de Confusión', y=1.1)
    plt.ylabel('Realidad')
    plt.xlabel('Predicción')

    TN = cnf_matrix[0,0]
    FP = cnf_matrix[0,1]
    FN = cnf_matrix[1,0]
    TP = cnf_matrix[1,1]

    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1 = 2*(Precision*Recall)/(Precision+Recall)
    TPR = Recall
    FPR = FP/(FP+TN)
       
    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=600, height=400
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show()
   
    AUC = auc(fpr, tpr)
   
    data = {"Accuracy":[Accuracy], "Precision":[Precision], "Recall":[Recall], "F1":[F1], "TPR":[TPR], "FPR":[FPR], "AUC":[AUC]}
    df = pd.DataFrame(data)
   
    return df

def metrics_clasif(model, X, y):
    
    y_pred=model.predict(X)
    y_score=model.predict_proba(X)[:,1]
    table = metrics_clasif_calc(y,y_pred,y_score)
    display(table)
    
    
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def metricas_regresion(df_x, df_y, model):
    
    y_pred=model.predict(df_x)
    y_true = df_y 
    n_obs = df_x.shape[1]
    n_x = df_x.shape[0]
    
    mae=mean_absolute_error(y_true,y_pred)
    mse=mean_squared_error(y_true,y_pred)
    rmse=np.sqrt(mse)
    r2=r2_score(y_true,y_pred)
    r2_adj=1-(((n_obs-1)/(n_obs-n_x-1)))*(1-r2)
    
    data = {"MAE":[mae], "MSE":[mse], "RMSE":[rmse], "R2":[r2], "R2_ADJ":[r2_adj]}
    df = pd.DataFrame(data)
    
    return df


def resultados(df,pipeline,tgt):
    df1 = df.copy()
    df2 = df.copy()
    df_re = df.copy()
    df_re["pred"]=pipeline.predict(df1)
    df_re["proba"]=pipeline.predict_proba(df2)[:,1]
    proba=[]
    no=[]
    yes=[]
    for i in [pd.Interval(-1,.1),pd.Interval(.1,.2),pd.Interval(.2,.3),pd.Interval(.3,.4),pd.Interval(.4,.5),pd.Interval(.5,.6),pd.Interval(.6,.7),pd.Interval(.7,.8),pd.Interval(.8,.9),pd.Interval(.9,1)]:
        aux={0:0,1:0}
        dictio=df_re[df_re["proba"].map(lambda x:x in i)][tgt].value_counts().to_dict()
        aux.update(dictio)
        no.append(aux[0])
        yes.append(aux[1])
        proba.append("("+str(round(i.left,1))+","+str(round(i.right,1))+"]")

    resultado=pd.DataFrame({"Proba":proba,"low_fare":no,"high_fare":yes})
    return resultado


from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score,StratifiedKFold

from sklearn.model_selection import RandomizedSearchCV
def random_search(X_train,y_train,estimator,param_grid):
    rs = RandomizedSearchCV(cv=StratifiedKFold(5),
                  verbose=True,
                  scoring='roc_auc',
                  estimator=estimator,
                  n_jobs=-1,
                  param_distributions=param_grid)
    rs.fit(X_train,y_train)
    print(f"Best Score : {rs.best_score_}")
    print(f"Best Params : {rs.best_params_}")
    return rs.best_estimator_


def plot_estab(X,y, model, lower_model, upper_model):
    df_aux=pd.DataFrame()
    df_aux["true"] =y
    df_aux["pred"] = model.predict(X)    
    df_aux["lower"] = lower_train = lower_model.predict(X)
    df_aux["upper"] = upper_train = upper_model.predict(X)
    df_aux["res"]=df_aux[["true","pred"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
    df_aux.reset_index(inplace = True, drop = True )
    df_aux[["true","pred", "lower", "upper"]].head(30).iplot()
    
    
def random_search2(X_train,y_train,estimator,param_grid):
    rs = RandomizedSearchCV(
                  verbose=True,
                  scoring='r2',
                  estimator=estimator,
                  n_jobs=-1,
                  param_distributions=param_grid)
    rs.fit(X_train,y_train)
    print(f"Best Score : {rs.best_score_}")
    print(f"Best Params : {rs.best_params_}")
    return rs.best_estimator_

def transformaciones_df(df_orig):
    df_orig["fare_class"] = df_orig["fare_class"].map({"low_fare": 0, "high_fare":1 })
    df_orig.query("fare_amount > 0 and pickup_longitude >= -180 and pickup_longitude <= 180 and \
                    dropoff_longitude >= -180 and dropoff_longitude <= 180 and \
                    pickup_latitude >= -90 and pickup_latitude <= 90 and \
                    dropoff_latitude >= -90 and dropoff_latitude <= 90 and passenger_count > 0  and \
                    passenger_count < 208", inplace = True)
    return df_orig