import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import shapiro, levene, normaltest, ttest_rel, ttest_ind, wilcoxon, mannwhitneyu
from lifelines import KaplanMeierFitter

matplotlib.use('TkAgg')


def load_dataframe(filename):
    df = pd.read_csv(filename)
    return df


def convert_minute(minute):
    if isinstance(minute, str) and '+' in minute:
        parts = minute.split('+')
        return int(parts[0]) + int(parts[1])
    return int(minute)


def kaplan_mayers(df):
    df = df[['Player', 'Minute']]
    df['Minute'] = df['Minute'].apply(convert_minute)

    ronaldo = df[df['Player'] == 'Cristiano Ronaldo'].sort_values(by='Minute').reset_index(drop=True)
    messi = df[df['Player'] == 'Lionel Messi'].sort_values(by='Minute').reset_index(drop=True)
    return ronaldo, messi


def print_kaplan_mayers(df1, df2):
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


piece_map = {
    'p': -1, 'P': 1,
    'n': -3, 'N': 3,
    'b': -3, 'B': 3,
    'r': -5, 'R': 5,
    'q': -9, 'Q': 9,
    'k': -100, 'K': 100,
}


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


def test_normal(df_s1, df_s2=None, col1: str = None, col2: str = None, fil: str = None,
                dependent: bool = True, max_size_diff_ratio=0.05):
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


def t_student(df1, df2=None, col1: str = None, col2: str = None, dependent: bool = True, param: bool = True):
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
