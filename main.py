import json
import numpy as np
from scipy.stats import friedmanchisquare
from utils import (
    load_dataframe,
    descriptive_statistics,
    test_normal,
    t_student,
    kaplan_mayers,
    print_kaplan_mayers,
    regression,
    roc, load_and_prepare, nemenyi_friedmann
)


### ZADANIE 1 ###
print('==================== Statystyka opisowa ====================')
df = load_dataframe(f'full_data.csv')
leagues = descriptive_statistics(df)

with open("league_stats.json", "w") as f:
    json.dump(leagues, f, indent=4)


### ZADANIE 2 ###
fifa_17 = load_dataframe(f'D:/datasets/fifa/FIFA17_official_data.csv')
fifa_17 = fifa_17.rename(columns={"Overall": "Overall17", "SprintSpeed": "SprintSpeed17"})[["ID", "Overall17", "SprintSpeed17"]]
fifa_18 = load_dataframe(f'D:/datasets/fifa/FIFA18_official_data.csv')
fifa_18 = fifa_18.rename(columns={"Overall": "Overall18", "SprintSpeed": "SprintSpeed18"})[["ID", "Overall18", "SprintSpeed18"]]
fifa_inner = fifa_17.join(fifa_18, on='ID', how='inner', lsuffix='_left', rsuffix='_right')
fifa_17 = fifa_inner.rename(columns={"Overall17": "Overall"})
fifa_18 = fifa_inner.rename(columns={"Overall18": "Overall"})
print(fifa_17)
print(fifa_18)

print('==================== t-Student dane zależne ====================')
if test_normal(fifa_17, fifa_18, 'Overall', dependent=True):
    t_student(fifa_17, fifa_18, 'Overall', dependent=True)

print('==================== t-Student dane niezależne ====================')

df = load_dataframe(f'full_data.csv')
if test_normal(df, col1='H_Attacks', col2='A_Attacks', fil='league-two', dependent=False):
    df_independent = df[df.League == 'league-two']
    t_student(df, col1='H_Attacks', col2='A_Attacks', dependent=False)


### ZADANIE 3 ###
print('==================== Wilcoxon dane zależne ====================')

fifa_17 = fifa_17.rename(columns={"SprintSpeed17": "SprintSpeed"})
fifa_18 = fifa_18.rename(columns={"SprintSpeed18": "SprintSpeed"})

if not test_normal(fifa_17, fifa_18, 'SprintSpeed', dependent=True):
    t_student(fifa_17, fifa_18, 'SprintSpeed', dependent=True, param=False)

print('==================== Mann-Whitney dane niezależne ====================')

if not test_normal(df, col1='H_Attacks', col2='A_Attacks', fil='bundesliga', dependent=False):
    df_independent = df[df.League == 'bundesliga']
    t_student(df, col1='H_Attacks', col2='A_Attacks', dependent=False, param=False)


### ZADANIE 4 ###
print('==================== Friedmann dane zależne ====================')
dfs = [load_and_prepare(f'D:/datasets/fifa/FIFA{year}_official_data.csv', year) for year in range(17, 23)]

fifa_inner = dfs[0]
for df in dfs[1:]:
    fifa_inner = fifa_inner.merge(df, on='ID', how='inner')

s, p = friedmanchisquare(fifa_inner['Finishing17'], fifa_inner['Finishing18'], fifa_inner['Finishing19'], fifa_inner['Finishing20'], fifa_inner['Finishing21'], fifa_inner['Finishing22'])
print(s, p)
if p < 0.05:
    nemenyi_friedmann(fifa_inner)


### ZADANIE 5 ###
# print('==================== Kaplan-Meyer ====================')
# df = load_dataframe('ronaldo_messi.csv')
# ronaldo, messi = kaplan_mayers(df)
# print_kaplan_mayers(ronaldo, messi)
#
#
# ### ZADANIE 6 ###
# print('==================== Regresja logistyczna ====================')
# df = load_dataframe('D:/datasets/chess_eval/chessData.csv')
# df = df.sample(n=10000)
# tprs, aucs, mean_fpr, accuracies = regression(df)
# print(f"Średnia dokładność: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
#
#
# ### ZADANIE 7 ###
# print('==================== krzywa ROC ====================')
# roc(tprs, aucs, mean_fpr)
