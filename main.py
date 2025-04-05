import json
import numpy as np
from utils import (
    load_dataframe,
    descriptive_statistics,
    test_normal,
    t_student,
    kaplan_mayers,
    print_kaplan_mayers,
    regression,
    roc
)


### ZADANIE 1 ###
df = load_dataframe(f'full_data.csv')
leagues = descriptive_statistics(df)

with open("league_stats.json", "w") as f:
    json.dump(leagues, f, indent=4)


### ZADANIE 2 ###
fifa_17 = load_dataframe(f'D:/datasets/fifa/FIFA17_official_data.csv')
fifa_18 = load_dataframe(f'D:/datasets/fifa/FIFA18_official_data.csv')

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

if not test_normal(fifa_17, fifa_18, 'SprintSpeed', dependent=True):
    t_student(fifa_17, fifa_18, 'SprintSpeed', dependent=True, param=False)

print('==================== Mann-Whitney dane niezależne ====================')

if not test_normal(df, col1='H_Attacks', col2='A_Attacks', fil='bundesliga', dependent=False):
    df_independent = df[df.League == 'bundesliga']
    t_student(df, col1='H_Attacks', col2='A_Attacks', dependent=False, param=False)


### ZADANIE 5 ###
df = load_dataframe('ronaldo_messi.csv')
ronaldo, messi = kaplan_mayers(df)
print_kaplan_mayers(ronaldo, messi)


### ZADANIE 6 ###
df = load_dataframe('D:/datasets/chess_eval/chessData.csv')
df = df.sample(n=10000)
tprs, aucs, mean_fpr, accuracies = regression(df)
print(f"Średnia dokładność: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")


### ZADANIE 7 ###
roc(tprs, aucs, mean_fpr)
