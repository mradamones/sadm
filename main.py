import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import shapiro, stats, kstest, normaltest, ttest_rel

from utils import (
    load_dataframe,
    kaplan_mayers,
    print_kaplan_mayers,
    fen_to_matrix, t_student, test_normal
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, auc

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
# df = load_dataframe('ronaldo_messi.csv')
# ronaldo, messi = kaplan_mayers(df)
# print_kaplan_mayers(ronaldo, messi)

### ZADANIE 6 ###
# df = load_dataframe('D:/datasets/chess_eval/chessData.csv')
# df = df.sample(n=100000)
# df["Evaluation"] = df["Evaluation"].astype(str).str.replace(r"[+#]", "", regex=True).astype(int)
# df["Class"] = (df["Evaluation"] >= 0).astype(int)
# df["Matrix"] = df["FEN"].apply(fen_to_matrix)
# df["Flattened"] = df["Matrix"].apply(lambda x: np.array(x).flatten().tolist())
#
# X = np.vstack(df["Flattened"])
# y = df["Class"].values
#
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
#
# clf = LogisticRegression(max_iter=1000)
#
# mean_fpr = np.linspace(0, 1, 100)
# tprs = []
# aucs = []
# accuracies = []
#
# for train_idx, test_idx in cv.split(X, y):
#     X_train, X_test = X[train_idx], X[test_idx]
#     y_train, y_test = y[train_idx], y[test_idx]
#
#     clf.fit(X_train, y_train)
#     y_proba = clf.predict_proba(X_test)[:, 1]
#     y_pred = clf.predict(X_test)
#
#     acc = accuracy_score(y_test, y_pred)
#     accuracies.append(acc)
#
#     fpr, tpr, _ = roc_curve(y_test, y_proba)
#     interp_tpr = np.interp(mean_fpr, fpr, tpr)
#     interp_tpr[0] = 0.0
#     tprs.append(interp_tpr)
#     aucs.append(auc(fpr, tpr))
#
# print(f"Średnia dokładność: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
#
# ### ZADANIE 7 ###
#
# mean_tpr = np.mean(tprs, axis=0)
# std_tpr = np.std(tprs, axis=0)
# mean_auc = np.mean(aucs)
# std_auc = np.std(aucs)
#
# plt.figure(figsize=(8, 6))
# plt.plot(mean_fpr, mean_tpr, color='blue',
#          label=f'Średnia ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2)
# plt.fill_between(mean_fpr,
#                  np.maximum(mean_tpr - std_tpr, 0),
#                  np.minimum(mean_tpr + std_tpr, 1),
#                  color='blue', alpha=0.2, label='±1 std')
#
# plt.plot([0, 1], [0, 1], 'k--', lw=1)
# plt.title('Średnia krzywa ROC (5-Fold CV)')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc='lower right')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
