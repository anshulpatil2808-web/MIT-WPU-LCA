import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target_names[iris.target]

# Display the dataset
print(df.head())

sns.barplot(data=df, x="species", y="sepal length (cm)")
plt.title("Sepal Length by Species")
plt.show()

sns.lineplot(x=df.index, y=df["sepal length (cm)"], hue=df["species"])
plt.title("Sepal Length Across Samples")
plt.show()

sns.scatterplot(
    data=df,
    x="sepal length (cm)",
    y="petal length (cm)",
    hue="species",
)
plt.title("Sepal Length and Petal Length")
plt.show()


sns.heatmap(df.iloc[:, :4].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlations")
plt.show()
