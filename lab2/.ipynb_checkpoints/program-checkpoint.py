import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from statistics import mean, variance

# -------- A1 --------
def load_purchase_data(filepath):
    df = pd.read_excel(filepath, sheet_name='Purchase Data')
    return df

def create_matrices_AXC(df):
    A = df.iloc[:, 1:-1].values  # purchase quantities
    C = df.iloc[:, -1].values.reshape(-1, 1)  # total payment
    return A, C

def get_vector_space_info(A):
    dimension = A.shape[1]
    vectors = A.shape[0]
    return dimension, vectors

def get_matrix_rank(A):
    return np.linalg.matrix_rank(A)

def get_product_costs(A, C):
    A_pinv = np.linalg.pinv(A)
    X = A_pinv @ C
    return X.flatten()

# -------- A2 --------
def classify_rich_poor(df):
    df['Class'] = df['TotalPayment'].apply(lambda x: 'RICH' if x > 200 else 'POOR')
    return df

# -------- A3 --------
def load_irctc_data(filepath):
    df = pd.read_excel(filepath, sheet_name='IRCTC Stock Price')
    return df

def calc_mean_variance_price(df):
    prices = df.iloc[:, 3].dropna()
    return mean(prices), variance(prices)

def filter_by_day(df, day):
    return df[df['Date'].dt.day_name() == day]

def filter_by_month(df, month_name):
    return df[df['Date'].dt.month_name() == month_name]

def prob_loss(df):
    return sum(df['Chg%'] < 0) / len(df['Chg%'])

def prob_profit_on_day(df, day):
    day_df = df[df['Date'].dt.day_name() == day]
    return sum(day_df['Chg%'] > 0) / len(day_df)

def conditional_prob_profit_given_day(df, day):
    total_day = df[df['Date'].dt.day_name() == day]
    return sum(total_day['Chg%'] > 0) / len(total_day)

def plot_chg_vs_day(df):
    df['Day'] = df['Date'].dt.day_name()
    sns.scatterplot(x='Day', y='Chg%', data=df)
    plt.title("Chg% vs Day")
    plt.xticks(rotation=45)
    plt.show()

# -------- A4 --------
def load_thyroid_data(filepath):
    df = pd.read_excel(filepath, sheet_name='thyroid0387_UCI')
    return df

def explore_data(df):
    types = df.dtypes
    missing = df.isnull().sum()
    stats = df.describe()
    return types, missing, stats

# -------- A5 --------
def jaccard_smc(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    jc = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) else 0
    smc = (f11 + f00) / (f11 + f00 + f01 + f10) if (f11 + f00 + f01 + f10) else 0
    return jc, smc

# -------- A6 --------
def cosine_sim(v1, v2):
    return cosine_similarity([v1], [v2])[0][0]

# -------- A7 --------
def heatmap_similarity(df):
    jc_matrix = np.zeros((20, 20))
    smc_matrix = np.zeros((20, 20))
    cos_matrix = np.zeros((20, 20))
    bin_df = df.select_dtypes(include=[np.number]).iloc[:20].applymap(lambda x: 1 if x > 0 else 0)

    for i in range(20):
        for j in range(20):
            jc_matrix[i, j], smc_matrix[i, j] = jaccard_smc(bin_df.iloc[i].values, bin_df.iloc[j].values)
            cos_matrix[i, j] = cosine_sim(df.iloc[i].values, df.iloc[j].values)

    sns.heatmap(jc_matrix, annot=False, cmap='Blues')
    plt.title('Jaccard Coefficient Heatmap')
    plt.show()

    sns.heatmap(smc_matrix, annot=False, cmap='Greens')
    plt.title('SMC Heatmap')
    plt.show()

    sns.heatmap(cos_matrix, annot=False, cmap='Oranges')
    plt.title('Cosine Similarity Heatmap')
    plt.show()

# -------- A8 --------
def impute_missing(df):
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            if df[col].isnull().sum() > 0:
                if abs(df[col].mean() - df[col].median()) > 1:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# -------- A9 --------
def normalize_data(df):
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# -------- Main --------
def main():
    filepath = "Lab Session Data.xlsx"

    # A1
    print("------ A1: Linear Algebra ------")
    df_purchase = load_purchase_data(filepath)
    A, C = create_matrices_AXC(df_purchase)
    dim, vecs = get_vector_space_info(A)
    print(f"Vector Space Dimensionality: {dim}")
    print(f"Number of Vectors: {vecs}")
    print(f"Rank of Matrix A: {get_matrix_rank(A)}")
    product_costs = get_product_costs(A, C)
    print(f"Product Costs (X): {product_costs}")

    # A2
    print("\n------ A2: Classify RICH vs POOR ------")
    df_classified = classify_rich_poor(df_purchase)
    print(df_classified[['TotalPayment', 'Class']].head())

    # A3
    print("\n------ A3: Stock Data Analysis ------")
    df_stock = load_irctc_data(filepath)
    df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')
    mean_price, var_price = calc_mean_variance_price(df_stock)
    print(f"Mean of Price (Population): {mean_price}")
    print(f"Variance of Price: {var_price}")
    wednesdays = filter_by_day(df_stock, "Wednesday")
    print(f"Mean of Price on Wednesdays: {mean(wednesdays.iloc[:, 3])}")
    april_data = filter_by_month(df_stock, "April")
    print(f"Mean of Price in April: {mean(april_data.iloc[:, 3])}")
    print(f"Probability of Loss: {prob_loss(df_stock):.4f}")
    print(f"Probability of Profit on Wednesday: {prob_profit_on_day(df_stock, 'Wednesday'):.4f}")
    print(f"Conditional Probability of Profit | Wednesday: {conditional_prob_profit_given_day(df_stock, 'Wednesday'):.4f}")
    plot_chg_vs_day(df_stock)

    # A4
    print("\n------ A4: Thyroid Data Exploration ------")
    df_thyroid = load_thyroid_data(filepath)
    types, missing, stats = explore_data(df_thyroid)
    print("Data Types:\n", types)
    print("Missing Values:\n", missing)
    print("Stats:\n", stats)

    # A5
    print("\n------ A5: Jaccard and SMC between first 2 vectors ------")
    bin_df = df_thyroid.select_dtypes(include=[np.number]).applymap(lambda x: 1 if x > 0 else 0)
    v1 = bin_df.iloc[0].values
    v2 = bin_df.iloc[1].values
    jc, smc = jaccard_smc(v1, v2)
    print(f"Jaccard Coefficient: {jc:.4f}")
    print(f"Simple Matching Coefficient: {smc:.4f}")

    # A6
    print("\n------ A6: Cosine Similarity ------")
    full_v1 = df_thyroid.select_dtypes(include=[np.number]).iloc[0].values
    full_v2 = df_thyroid.select_dtypes(include=[np.number]).iloc[1].values
    cos_sim = cosine_sim(full_v1, full_v2)
    print(f"Cosine Similarity: {cos_sim:.4f}")

    # A7
    print("\n------ A7: Heatmap of Similarity Measures ------")
    heatmap_similarity(df_thyroid)

    # A8
    print("\n------ A8: Missing Value Imputation ------")
    df_imputed = impute_missing(df_thyroid.copy())
    print("Missing values after imputation:\n", df_imputed.isnull().sum())

    # A9
    print("\n------ A9: Data Normalization ------")
    df_normalized = normalize_data(df_imputed.copy())
    print("First 5 rows after normalization:\n", df_normalized.head())

# Run the full lab
main()
