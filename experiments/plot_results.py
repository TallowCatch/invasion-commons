import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("results/sweep.csv")

    # collapse rate
    collapse_rate = df["collapsed"].mean()
    print("Collapse rate:", collapse_rate)

    # simple hist
    plt.figure()
    df["mean_stock"].hist(bins=30)
    plt.title("Mean stock over episodes")
    plt.xlabel("mean stock")
    plt.ylabel("count")
    plt.show()

    plt.figure()
    df["payoff_gini"].hist(bins=30)
    plt.title("Payoff inequality (Gini)")
    plt.xlabel("gini")
    plt.ylabel("count")
    plt.show()


if __name__ == "__main__":
    main()