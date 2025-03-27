from utils import (
    load_dataframe,
    kaplan_mayers,
    print_kaplan_mayers,
    fen_to_matrix
)
from sklearn.linear_model import LogisticRegression

### ZADANIE 5 ###
# df = load_dataframe('ronaldo_messi.csv')
# ronaldo, messi = kaplan_mayers(df)
# print_kaplan_mayers(ronaldo, messi)

### ZADANIE 6 ###
df = load_dataframe('./chess_eval/chessData.csv')
df["Evaluation"] = df["Evaluation"].astype(str).str.replace(r"[+#]", "", regex=True).astype(int)

df["Class"] = (df["Evaluation"] >= 0).astype(int)
df = df[['FEN', 'Class']]
df["Matrix"] = df["FEN"].apply(fen_to_matrix)
df["Flattened"] = df["Matrix"].apply(lambda x: x.flatten().tolist())
print(df)
X = df['Flattened']
y = df['Class']
clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X[:2, :])
clf.predict_proba(X[:2, :])
clf.score(X, y)
