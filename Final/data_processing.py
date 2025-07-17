import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


def convert(data):
    encoder = preprocessing.LabelEncoder()
    data["theme"] = encoder.fit_transform(data["theme"])
    return data.fillna(-999)


def plot_trendline(x, y, degree, ylabel):
    coeffs = np.polyfit(x, y, degree)
    poly_fn = np.poly1d(coeffs)

    plt.figure()
    plt.plot(x, y, "ko", label="Data")
    plt.plot(x, poly_fn(x), "r--", label=f"Degree {degree}")
    plt.xlabel("Movie Rating")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def preprocess_csv(file):
    df = pd.read_csv(file)

    # --- Process actors ---
    actors = df["Star"].str.split(",", expand=True).iloc[:, :2]
    actors = actors.apply(
        lambda col: col.str.replace(r"[\[\]']", "", regex=True).str.strip())
    df.insert(9, "actor1", actors[0])
    df.insert(10, "actor2", actors[1])
    df.drop(columns="Star", inplace=True)

    # --- Process theme ---
    theme = df["theme"].str.split(",", expand=True)[0].str.replace("\n",
                                                                   "",
                                                                   regex=True)
    df.drop(columns="theme", inplace=True)
    df.insert(4, "theme", theme)

    # --- Remove index column if exists ---
    if df.columns[0].lower() in ["unnamed: 0", "id", "index"]:
        df.drop(columns=df.columns[0], inplace=True)

    # --- Clean year, watchtime, votes, gross ---
    df["Year of relase"] = df["Year of relase"].str.replace(
        "I", "", regex=True).str.strip().astype(int)
    df["Watchtime"] = df["Watchtime"].astype(int)
    df["Votes"] = df["Votes"].str.replace(",", "", regex=True).astype(int)

    df = df[df["Gross collection"] != "*****"]
    df["Gross collection($M)"] = df["Gross collection"] \
        .str.replace("[#$M]", "", regex=True).astype(float)
    df.drop(columns="Gross collection", inplace=True)

    # --- Clean rating ---
    df["Movie Rating"] = df["Movie Rating"].astype(float)

    # --- Drop unnecessary columns ---
    df.drop(columns=["Name of movie", "Director", "actor1", "actor2"],
            inplace=True)

    return df


if __name__ == "__main__":
    file_path = "Top_600_IMDB_Movies.csv"
    processed_path = "Top_600_IMDB_Movies_processed.csv"

    df = preprocess_csv(file_path)
    df = convert(df)

    # Save processed file
    df.to_csv(processed_path, index=False, encoding="utf-8-sig")

    # Plot trends
    plot_trendline(df["Movie Rating"], df["Year of relase"], 3,
                   "Year of Release")
    plot_trendline(df["Movie Rating"], df["Watchtime"], 5, "Watchtime (min)")
    plot_trendline(df["Movie Rating"], df["theme"], 3, "Theme (encoded)")
    plot_trendline(df["Movie Rating"], df["Votes"], 5, "Votes")
    plot_trendline(df["Movie Rating"], df["Gross collection($M)"], 5,
                   "Gross Collection ($M)")

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True),
                annot=True,
                cmap="YlOrRd",
                fmt=".2f")
    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.show()
