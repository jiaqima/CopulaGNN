from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_data

np.random.seed(0)

data = load_data()
features = data["x"].numpy()
labels = data["y"].numpy()
embeddings = TruncatedSVD(n_components=50).fit_transform(features)
embeddings = TSNE(n_components=1).fit_transform(embeddings)
embeddings = np.squeeze(embeddings)

df = pd.DataFrame({"labels": labels, "embeddings": embeddings}).groupby("labels").median().reset_index().sort_values("embeddings")
print(df)

plt.scatter(x=embeddings, y=np.zeros_like(labels), c=labels, alpha=0.1)
plt.scatter(x=df.embeddings, y=np.ones_like(df.labels), c=df.labels)
for emb, label in zip(df.embeddings, df.labels):
    plt.text(emb, y=0.95, s=str(label))
plt.savefig("embedding.png")
plt.close()