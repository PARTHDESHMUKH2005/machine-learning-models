import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import zscore


def main():
    # Load dataset
    df = pd.read_csv("data/sales_data.csv")

    # Outlier detection using Z-score
    df["z_score"] = zscore(df["Sales_Revenue"])
    df_clean = df[df["z_score"].abs() < 3]

    # Features and target
    X = df_clean[["Advertising_Spend"]]
    y = df_clean["Sales_Revenue"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

    # Visualization
    plt.figure()
    plt.scatter(X_test, y_test, label="Actual")
    plt.plot(X_test, y_pred, color="red", label="Predicted")
    plt.xlabel("Advertising Spend")
    plt.ylabel("Sales Revenue")
    plt.title("Sales Prediction using Linear Regression")
    plt.legend()
    plt.savefig("outputs/regression_plot.png")
    plt.show()

    # Residual plot
    residuals = y_test - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color="red")
    plt.xlabel("Predicted Sales")
    plt.ylabel("Residuals")
    plt.title("Residual Analysis")
    plt.savefig("outputs/residual_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
