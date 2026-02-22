import pandas as pd
import numpy as np
import os

rows = []
for label in range(10):
    for i in range(500):
        if label == 0:
            base = [10, 10, 10, 10, 10, 10, 10, 10]
        else:
            base = [label * 25 + np.random.randint(0, 30) for _ in range(8)]
        row = [0, label] + [max(0, v + np.random.randn() * 10) for v in base]
        rows.append(row)

cols = ["timestamp", "label"] + ["ch" + str(i) for i in range(8)]
df = pd.DataFrame(rows, columns=cols)
os.makedirs("data", exist_ok=True)
df.to_csv("data/test_session.csv", index=False)
print("Done! " + str(len(df)) + " samples saved.")