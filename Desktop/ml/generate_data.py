import numpy as np
import pandas as pd

def generate_sales_data(
    n_samples=100,
    min_ad_spend=1000,
    max_ad_spend=50000,
    random_state=42
):
    """
    Generates synthetic advertising spend and sales revenue data
    """

    np.random.seed(random_state)

    # Advertising spend (independent variable)
    advertising_spend = np.random.uniform(
        min_ad_spend, max_ad_spend, n_samples
    )

    # Sales revenue (dependent variable)
    # Base sales + linear effect of ads + random noise
    sales_revenue = (
        5000 +
        2.5 * advertising_spend +
        np.random.normal(0, 10000, n_samples)
    )

    df = pd.DataFrame({
        "Advertising_Spend": advertising_spend,
        "Sales_Revenue": sales_revenue
    })

    return df


if __name__ == "__main__":
    df = generate_sales_data()
    df.to_csv("data/sales_data.csv", index=False)
    print("Synthetic sales dataset generated and saved to data/sales_data.csv")
