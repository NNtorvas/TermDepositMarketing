import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
# from IPython import display


def get_percent(df_percent, mask_percent, df_total, mask_total):
    return np.round((df_percent.loc[mask_percent,:].shape[0] / df_total.loc[mask_total,:].shape[0]) * 100, 2)


def plot_barplot_with_percent(df, col, labels, figsize=(4,4)):
    
    df1 = df[col].value_counts(normalize=True)
    df1 = df1.mul(100)

    fig, ax = plt.subplots(1,1, figsize=figsize)
    sns.barplot(x=df1.index, y=df1.values, orient='v', ax=ax).set(xlabel=labels['xlabel'], title=labels['title'])
    ax.bar_label(ax.containers[0], fmt='%.2f%%');


def get_df_x_vs_y(df, x):
    
    Y0_mask = df['Y'] == 'no'
    Y1_mask = df['Y'] == 'yes'
    
    df_x = pd.DataFrame(index = [0,1])
    for i, clas in enumerate(df[x].unique()):
        df_x[clas] = [df[Y0_mask & (df[x] == clas)].shape[0], df[Y1_mask & (df[x] == clas)].shape[0]]

    return df_x


def plot_heatmap_with_target_dist(df, feat, figsize=(5,5)):
    df_x = get_df_x_vs_y(df, feat)
    df_x = df_x.apply(lambda x: (x/df.shape[0]))
    
    fig = plt.figure(figsize=figsize)
    ax = sns.heatmap(df_x, annot=True, fmt='.2%').set(xlabel=feat, ylabel='Y')


def plot_distribution_with_stats(df, col, graph=True, lines=True, kde=True):
    bp = plt.boxplot(df[col], vert=False, widths=5);
    plt.close()
    stats = dict()
    for key in ['caps', 'boxes', 'medians']:
        stats[key] = [item.get_xdata() for item in bp[key]]

    aux = pd.DataFrame({'min': df[col].min(),
                        'left cap': stats['caps'][0][0], 
                        'Q1': stats['boxes'][0][0],
                        'Q2': stats['medians'][0][0],
                        'Q3': stats['boxes'][0][2],
                        'right cap': stats['caps'][1][0],
                        'max': df[col].max(),
                        'mean': df[col].mean(),
                        'mode': df[col].mode()
                        }, index=[0])
    
    if graph:
        title = col[0].upper()+col[1::]
        fig, ax = plt.subplots(1,1, figsize=(10,3))
        fig.suptitle(title)

        sns.histplot(df[col], kde=kde, ax=ax).set(xlabel=title, ylabel='frequency');

        if lines:
            plt.axvline(aux.loc[0,'min'], 0,1, color='black', linestyle='--');
            plt.axvline(aux.loc[0,'left cap'], 0,1, color='red', linestyle='--');
            plt.axvline(aux.loc[0,'Q1'], 0,1, color='green', linestyle='--');
            plt.axvline(aux.loc[0,'Q2'], 0,1, color='green', linestyle='--');
            plt.axvline(aux.loc[0,'mean'], 0,1, color='purple', linestyle='--');
            plt.axvline(aux.loc[0,'Q3'], 0,1, color='green', linestyle='--');
            plt.axvline(aux.loc[0,'right cap'], 0,1, color='red', linestyle='--');
            plt.axvline(aux.loc[0,'max'], 0,1, color='black', linestyle='--');

            plt.legend(['distribution', 'min', 'left cap', 'Q1', 'Q2', 'mean', 'Q3', 'right cap', 'max'], bbox_to_anchor=(1.01, 1));

        plt.show()

    if lines: display(aux)

    return aux


def stat_analysis(df, target):
    y = df[target]
    y_type = y.dtypes
    # https://towardsdatascience.com/statistics-in-python-using-anova-for-feature-selection-b4dc876ef4f0
    x = df.drop(columns=target)

    x_int, x_cat = [], []
    [x_int.append(col) if x[col].dtype == 'int' else x_cat.append(col) for col in x.columns]

    if y_type == 'object':  # target is categorical
        get_chi_square_test(df, target, x_cat)  # for categorical variables        
        get_anova_test(df, target, x_int)       # for numerical variables
    
    else:   # target is numerical
        # get_t_test(df, target, x_int)           # for numerical variables
        pass

    get_correlation_matrix(df, x_int, (11,5))


def get_correlation_matrix(df, cols, figsize):
    print('\nCORRELATION MATRIX BETWEEN NUMERICAL ATTRIBUTES')
    fig = plt.figure(figsize=figsize)
    ax = sns.heatmap(df[cols].corr(), annot=True, vmin=-1, cmap="YlGnBu", linewidths=.5)
    ax.set_title("Correlation between numeric attributes");


def get_chi_square_test(df, target, cols):
    
    print('CHI SQUARE TEST BETWEEN CATEGORICAL ATTRIBUTES')
    print('If feature combination is not printed, features are NOT correlated.\n')
    
    print('--> Target vs features')
    for f in cols: chi_square_test_results(df, target, f)

    print('\n--> Between features')
    for i, f1 in enumerate(cols[1::]): 
        i +=1
        for f2 in cols[i::]:
            if f1 != f2: chi_square_test_results(df, f1, f2)


def chi_square_test_results(df, feat1, feat2, alpha=0.05):
    
    new_test_df = pd.crosstab(index=df[feat2], columns=df[feat1],margins=True)
    stat, p, dof, _= chi2_contingency(new_test_df)

    if p < alpha: print(f'[{feat1}-{feat2}] p-value: {np.round(p, 5)} --> result IS significant. Features are correlated.')
    else: pass # print(f'[{feat1}-{feat2}] p-value: {np.round(p, 3)} --> result IS NOT significant. Features are NOT correlated.')


def get_anova_test(df, target, cols):
    
    print('\nANOVA TEST FOR NUMERICAL ATTRIBUTES AGAINST TARGET VARIABLE')
    print('If feature is not printed, we FAIL to reject the Null Hypothesis.')
    # https://towardsdatascience.com/anova-test-with-python-cfbf4013328b
    
    df[target] = df[target].astype(str)
    
    for col in cols:

        model = ols(col + '~' + target, data = df).fit() #Oridnary least square method
        result_anova = sm.stats.anova_lm(model) # ANOVA Test
        p_val = result_anova["PR(>F)"][0]

        alpha = 0.05
        # possible types "right-tailed, left-tailed, two-tailed"
        tail_hypothesis_type = "two-tailed"
        if tail_hypothesis_type == "two-tailed": alpha /= 2
        
        # The p-value approach
        if p_val <= alpha:
            print("--------------------------------------------------------------------------------------")
            print(f"{col.upper()}")
            print("The p-value approach to hypothesis testing in the decision rule")
            print(f"p value {p_val} <= alpha {alpha} --> Null Hypothesis is rejected.")

    return result_anova
    

def get_num_cat_columns(df):
    num_cols, cat_cols = [], []
    [num_cols.append(col) if df[col].dtype == 'int' else cat_cols.append(col) for col in df.columns[0:-1]];
    return num_cols, cat_cols


def one_hot_encode(df, target, cols):
    encoder_categories = []
    for col in cols:    
        col_categories = df[col].unique()
        encoder_categories.append(col_categories)

    encoder = OneHotEncoder(categories=encoder_categories, sparse=False, drop='first')
    encoder = encoder.fit(df[cols])
    X_encoder = encoder.transform(df[cols])

    X_categorical = pd.DataFrame(X_encoder, columns = encoder.get_feature_names_out(cols))
    X = pd.concat([df, X_categorical], axis=1)
    X = X.drop(columns=cols)
    X = X.drop(columns=target)

    return X


def train_val_test_split(X, df, target, test_size=0.2, val_size=0.2, partitions=2):

    if partitions == 2:
        X_train, X_test, y_train, y_test = train_test_split(X, df[target], stratify = df[target], test_size=test_size, shuffle=True, random_state=42)

        print('X_train', X_train.shape, '\tX_test',  X_test.shape)
        print('y_train', y_train.shape, '\ty_test',  y_test.shape)

        return (X_train, y_train), (X_test, y_test)
    
    elif partitions == 3:
        X_train, X_test, y_train, y_test = train_test_split(X, df[target], stratify = df[target], test_size=test_size, shuffle=True, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)

        print('X_train', X_train.shape, '\tX_val', X_val.shape, '\tX_test',  X_test.shape)
        print('y_train', y_train.shape, '\ty_val', y_val.shape, '\t\ty_test',  y_test.shape)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    else:
        print('Partitions should be 2 for train/test split or 3 for train/val/test split')


def test_multiple_models(model, model_name, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="yes")
    report = classification_report(y_test, y_pred)

    return [model_name, acc, prec, report]


def model_CV_grid_search (model, parameter_space, X_train, y_train, best_set=False, cols=[]):

    if cols == []:
        cols = X_train.columns

    # Define the cross validation method with StratifiedKFold
    folds = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    # Define search method (grid search) and evaluation métric (accuracy)
    grid = GridSearchCV(estimator=model, param_grid=parameter_space, 
                        cv=folds, scoring='accuracy', n_jobs=4)

    # Fit model
    grid.fit(X_train[cols], y_train)

    # Show the set of parameters with the best score
    if best_set:
        print("Best set of parameters: \n ", grid.best_params_)
        print("\nCV best accuracy: {}%\n".format((grid.best_score_*100).round(2)))
    
    return grid
    
    
def evaluate_model(model, model_name, X_train, y_train, X_test, y_test, cols=[]):
    
    if cols == []:
        cols = X_train.columns

    y_pred_train = model.best_estimator_.predict(X_train[cols])
    y_pred_test = model.best_estimator_.predict(X_test[cols])
    
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    report_train = classification_report(y_train, y_pred_train, zero_division=0)
    report_test = classification_report(y_test, y_pred_test, zero_division=0)
    
    print("Training accuracy = {}%".format((acc_train*100).round(2)))
    print("Testing accuracy = {}%".format((acc_test*100).round(2)))
    
    return [model_name, acc_train, acc_test, report_train, report_test]


