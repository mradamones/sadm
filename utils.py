import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


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
