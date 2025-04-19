import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scikit_posthocs as sp
from pandas import DataFrame
from scipy.stats import shapiro, levene, normaltest, ttest_rel, ttest_ind, wilcoxon, mannwhitneyu
from lifelines import KaplanMeierFitter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import RepeatedStratifiedKFold

matplotlib.use('TkAgg')
piece_map = {
    'p': -1, 'P': 1,
    'n': -3, 'N': 3,
    'b': -3, 'B': 3,
    'r': -5, 'R': 5,
    'q': -9, 'Q': 9,
    'k': -100, 'K': 100,
}


def load_dataframe(filename: str) -> DataFrame:
    df = pd.read_csv(filename)
    return df


def convert_minute(minute) -> int:
    if isinstance(minute, str) and '+' in minute:
        parts = minute.split('+')
        return int(parts[0]) + int(parts[1])
    return int(minute)


def kaplan_mayers(df: DataFrame):
    df = df[['Player', 'Minute']].copy()
    df['Minute'] = df['Minute'].apply(convert_minute)

    ronaldo = df[df['Player'] == 'Cristiano Ronaldo'].sort_values(by='Minute').reset_index(drop=True)
    messi = df[df['Player'] == 'Lionel Messi'].sort_values(by='Minute').reset_index(drop=True)
    return ronaldo, messi


def print_kaplan_mayers(df1: DataFrame, df2: DataFrame):
    kmf = KaplanMeierFitter()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 1 rząd, 2 kolumny

    kmf.fit(durations=df1['Minute'], event_observed=[1] * len(df1), label='Cristiano Ronaldo')
    kmf.plot_survival_function(ax=ax[0], ci_show=True)
    ax[0].set_title("Kaplan-Meier dla Cristiano Ronaldo")
    ax[0].set_xlabel("Minuta")
    ax[0].set_ylabel("Prawdopodobieństwo przeżycia")
    ax[0].grid(visible=True)

    kmf.fit(durations=df2['Minute'], event_observed=[1] * len(df2), label='Lionel Messi')
    kmf.plot_survival_function(ax=ax[1], ci_show=True)
    ax[1].set_title("Kaplan-Meier dla Lionela Messiego")
    ax[1].set_xlabel("Minuta")
    ax[1].set_ylabel("Prawdopodobieństwo przeżycia")
    ax[1].grid(visible=True)

    plt.tight_layout()
    plt.show()


def fen_to_matrix(fen):
    board = []
    rows = fen.split()[0].split('/')

    for row in rows:
        expanded_row = []
        for char in row:
            if char.isdigit():
                expanded_row.extend([0] * int(char))
            else:
                expanded_row.append(piece_map.get(char, 0))
        board.append(expanded_row)

    return np.array(board)


def test_normal(df_s1: DataFrame, df_s2: DataFrame = None, col1: str = None, col2: str = None, fil: str = None,
                dependent: bool = True, max_size_diff_ratio: float = 0.05):
    if dependent:
        df1 = df_s1[col1].dropna()
        df2 = df_s2[col1].dropna()
        stat1, p_norm1 = normaltest(df1)
        stat2, p_norm2 = normaltest(df2)
        len1, len2 = len(df1), len(df2)
        size_diff_ratio = abs(len1 - len2) / min(len1, len2)

        equal_length = size_diff_ratio <= max_size_diff_ratio
    else:
        df1 = df_s1[df_s1.League == fil][col1].dropna()
        df2 = df_s1[df_s1.League == fil][col2].dropna()
        stat1, p_norm1 = shapiro(df1)
        stat2, p_norm2 = shapiro(df2)
        equal_length = True

    stat_var, p_var = levene(df1, df2)

    normal = p_norm1 > 0.05 and p_norm2 > 0.05
    equal_var = p_var > 0.05
    if normal and equal_var and equal_length:
        print("Dane spełniają założenia testu t-Studenta (rozkład normalny, równe wariancje i równoliczność zbiorów)")
    else:
        if not normal:
            print("Co najmniej jedna z grup nie ma rozkładu normalnego")
        if not equal_var:
            print("Wariancje są różne")
        if not equal_length:
            print("Występuje różnoliczność zbiorów")

    return normal and equal_var and equal_length


def t_student(df1: DataFrame, df2: DataFrame = None, col1: str = None, col2: str = None, dependent: bool = True, param: bool = True):
    if df2 is None:
        df1_vals = df1[col1].dropna().reset_index(drop=True)
        df2_vals = df1[col2].dropna().reset_index(drop=True)
    else:
        df1_vals = df1[col1].dropna().reset_index(drop=True)
        df2_vals = df2[col2 if col2 else col1].dropna().reset_index(drop=True)

    min_len = min(len(df1_vals), len(df2_vals))
    df1_vals = df1_vals.iloc[:min_len]
    df2_vals = df2_vals.iloc[:min_len]
    if dependent:
        if param:
            stat, p = ttest_rel(df1_vals, df2_vals)
        else:
            res = wilcoxon(df1_vals, df2_vals)
            stat, p = res.statistic, res.pvalue
    else:
        if param:
            stat, p = ttest_ind(df1_vals, df2_vals, equal_var=False)
        else:
            stat, p = mannwhitneyu(df1_vals, df2_vals)

    print(f"t = {stat:.4f}, p = {p:.4f}")

    if p < 0.05:
        print("Różnica jest istotna statystycznie")
    else:
        print("Brak istotnej różnicy")

    return p < 0.05


def descriptive_statistics(df: DataFrame):
    stats = {}
    for league, group in df.groupby("League"):
        cols = ['H_Score', 'A_Score', 'HT_H_Score', 'HT_A_Score']
        res = {}
        for col in cols:
            avg = {}
            avg.update({f'{col}_avg': round(group[col].mean(), 2)})
            avg.update({f'{col}_mode': group[col].mode().tolist()})
            avg.update({f'{col}_median': round(group[col].median(), 2)})
            avg.update({f'{col}_range': round(group[col].max() - group[col].min(), 2)})
            avg.update({f'{col}_std': round(group[col].std(), 2)})
            avg.update({f'{col}_variance': round(group[col].var(), 2)})
            avg.update({f'{col}_cv': round(group[col].std() / group[col].mean(), 2)})
            avg.update({f'{col}_0.25': round(group[col].quantile(0.25), 0)})
            avg.update({f'{col}_0.50': round(group[col].quantile(0.50), 0)})
            avg.update({f'{col}_0.75': round(group[col].quantile(0.75), 0)})
            res.update({col: avg})
        stats.update({league: res})
    return stats


def regression(df: DataFrame):
    df["Evaluation"] = df["Evaluation"].astype(str).str.replace(r"[+#]", "", regex=True).astype(int)
    df["Class"] = (df["Evaluation"] >= 0).astype(int)
    df["Matrix"] = df["FEN"].apply(fen_to_matrix)
    df["Flattened"] = df["Matrix"].apply(lambda x: np.array(x).flatten().tolist())

    X = np.vstack(df["Flattened"])
    y = df["Class"].values

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)

    clf = LogisticRegression(max_iter=1000)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    accuracies = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
    return tprs, aucs, mean_fpr, accuracies


def roc(tprs, aucs, mean_fpr):
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=f'Średnia ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2)
    plt.fill_between(mean_fpr,
                     np.maximum(mean_tpr - std_tpr, 0),
                     np.minimum(mean_tpr + std_tpr, 1),
                     color='blue', alpha=0.2, label='±1 std')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.title('Średnia krzywa ROC (5-Fold CV)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def draw_cd_diagram(avg_ranks, labels, cd):
    plt.figure(figsize=(10, 2))
    y = 0.5

    for i, (rank, label) in enumerate(sorted(zip(avg_ranks, labels))):
        plt.plot(rank, y, 'o', markersize=8, color='black')
        plt.text(rank, y + 0.05, label, ha='center', fontsize=10, rotation=45)

    max_rank = max(avg_ranks)
    min_rank = min(avg_ranks)
    plt.hlines(y - 0.1, min_rank, max_rank, color='lightgray')
    plt.hlines(y - 0.2, min_rank, min_rank + cd, color='black', linewidth=2)
    plt.vlines([min_rank, min_rank + cd], y - 0.25, y - 0.15, color='black')
    plt.text(min_rank + cd / 2, y - 0.35, f"CD = {cd:.2f}", ha='center')

    plt.title("CD Diagram")
    plt.yticks([])
    plt.xlabel("Średnia ranga")
    plt.tight_layout()
    plt.show()


def load_and_prepare(path, year):
    df = load_dataframe(path)
    df = df.rename(columns={"Finishing": f"Finishing{year}"})[["ID", f"Finishing{year}"]]
    return df


def nemenyi_friedmann(fifa_inner):
    finishing_data = fifa_inner[[
        'Finishing17', 'Finishing18', 'Finishing19',
        'Finishing20', 'Finishing21', 'Finishing22'
    ]]

    nemenyi = sp.posthoc_nemenyi_friedman(finishing_data)
    print(nemenyi)

    sns.heatmap(nemenyi, annot=True, cmap='coolwarm', fmt=".3f")
    plt.title("Test Nemenyiego – porównanie lat FIFA")
    plt.show()

    k = 6
    N = len(fifa_inner)
    q_alpha = 2.850

    CD = q_alpha * np.sqrt((k * (k + 1)) / (6 * N))
    print(f"Wartość krytyczna (CD) dla testu Nemenyiego: {CD:.4f}")
    alpha = 0.05
    significant_pairs = []

    for i in range(len(nemenyi.columns)):
        for j in range(i + 1, len(nemenyi.columns)):
            p_value = nemenyi.iloc[i, j]
            if p_value < alpha:
                year1 = nemenyi.columns[i]
                year2 = nemenyi.columns[j]
                significant_pairs.append((year1, year2, p_value))

    print("Istotne różnice między latami (p < 0.05):")
    for pair in significant_pairs:
        print(f"{pair[0]} vs {pair[1]}: p = {pair[2]:.4g}")
    avg_ranks = finishing_data.rank(axis=1, method='average', ascending=False).mean().values
    labels = ['FIFA17', 'FIFA18', 'FIFA19', 'FIFA20', 'FIFA21', 'FIFA22']
    draw_cd_diagram(avg_ranks, labels, CD)