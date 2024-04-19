import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns


# Define Cohen's d function
def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(
        ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof
    )


# Load the dataset
df = pd.read_csv(r"\Data\ImageRank_detailed.csv")


# Preprocess the data
variation_1_split = df["Variation 1 Name"].str.split("|", expand=True)
variation_2_split = df["Variation 2 Name"].str.split("|", expand=True)
df_var1 = df[["Participant", "Pair", "Image Rank 1 Result"]].copy()
df_var2 = df[["Participant", "Pair", "Image Rank 2 Result"]].copy()
df_var1[["Colour", "Imagery", "Typography"]] = variation_1_split
df_var2[["Colour", "Imagery", "Typography"]] = variation_2_split
df_var1.rename(columns={"Image Rank 1 Result": "Result"}, inplace=True)
df_var2.rename(columns={"Image Rank 2 Result": "Result"}, inplace=True)
df_combined = pd.concat([df_var1, df_var2], ignore_index=True)


# Perform the ANOVA with interaction terms and visualise interactions
model = ols("Result ~ C(Colour) * C(Imagery) * C(Typography)", data=df_combined).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)
anova_results.to_csv("anova_results.csv")

sns.pointplot(data=df_combined, x="Colour", y="Result", hue="Imagery", errorbar=None)
plt.title("")
plt.show()
sns.pointplot(data=df_combined, x="Colour", y="Result", hue="Typography", errorbar=None)
plt.title("")
plt.show()
sns.pointplot(data=df_combined, x="Imagery", y="Result", hue="Typography", errorbar=None)
plt.title("")
plt.show()


# Tukey's HSD tests and plotting results
df_tukey = []
variables = ["Colour", "Imagery", "Typography"]
for variable in variables:
    tukey = pairwise_tukeyhsd(
        endog=df_combined["Result"], groups=df_combined[variable], alpha=0.05
    )
    print(f"\nTukey's HSD Test for {variable}:")
    print(tukey)
    tukey_results = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    tukey_results.to_csv(f"tukey_results_{variable}.csv")
    df_tukey.append(pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0]))
    tukey.plot_simultaneous()
    plt.title('')    
    plt.show()



df_tukey = pd.concat(df_tukey, ignore_index=True)
df_tukey.to_csv(f"tukey_results_overview.csv")
print(df_tukey)


# Cohen's d calculations for Colour
group_green = df_combined[df_combined["Colour"] == "Green"]["Result"]
group_blue = df_combined[df_combined["Colour"] == "Blue"]["Result"]
group_red = df_combined[df_combined["Colour"] == "Red"]["Result"]

d_green_blue = cohens_d(group_green, group_blue)
d_green_red = cohens_d(group_green, group_red)
d_blue_red = cohens_d(group_blue, group_red)

print(f"Cohen's d for Green vs. Blue: {d_green_blue}")
print(f"Cohen's d for Green vs. Red: {d_green_red}")
print(f"Cohen's d for Blue vs. Red: {d_blue_red}")

# Cohen's d calculations for Imagery
group_illustration = df_combined[df_combined["Imagery"] == "Illustration"]["Result"]
group_image = df_combined[df_combined["Imagery"] == "Image"]["Result"]
group_none = df_combined[df_combined["Imagery"] == "None"]["Result"]

d_illustration_image = cohens_d(group_illustration, group_image)
d_illustration_none = cohens_d(group_illustration, group_none)
d_image_none = cohens_d(group_image, group_none)

print(f"\nCohen's d for Illustration vs. Image: {d_illustration_image}")
print(f"Cohen's d for Illustration vs. None: {d_illustration_none}")
print(f"Cohen's d for Image vs. None: {d_image_none}")

# Cohen's d calculations for Typography
group_handwritten = df_combined[df_combined["Typography"] == "Handwritten"]["Result"]
group_sansserif = df_combined[df_combined["Typography"] == "Sans-Serif"]["Result"]
group_serif = df_combined[df_combined["Typography"] == "Serif"]["Result"]

d_handwritten_sansserif = cohens_d(group_handwritten, group_sansserif)
d_handwritten_serif = cohens_d(group_handwritten, group_serif)
d_sansserif_serif = cohens_d(group_sansserif, group_serif)

print(f"\nCohen's d for Handwritten vs. Sans-Serif: {d_handwritten_sansserif}")
print(f"Cohen's d for Handwritten vs. Serif: {d_handwritten_serif}")
print(f"Cohen's d for Sans-Serif vs. Serif: {d_sansserif_serif}")



# Function to calculate Cohen's d and return a DataFrame
def calculate_cohens_d(df, group1, group2, category):
    d = cohens_d(
        df[df[category] == group1]["Result"], df[df[category] == group2]["Result"]
    )
    return pd.DataFrame(
        {
            "Group1": [group1],
            "Group2": [group2],
            "Cohen's d": [d],
            "Category": [category],
        }
    )


# Prepare the data
categories = ["Colour", "Imagery", "Typography"]
comparisons = {
    "Colour": [("Green", "Blue"), ("Green", "Red"), ("Blue", "Red")],
    "Imagery": [("Illustration", "Image"), ("Illustration", "None"), ("Image", "None")],
    "Typography": [
        ("Handwritten", "Sans-Serif"),
        ("Handwritten", "Serif"),
        ("Sans-Serif", "Serif"),
    ],
}

all_results = []
for category, pairs in comparisons.items():
    for group1, group2 in pairs:
        all_results.append(calculate_cohens_d(df_combined, group1, group2, category))

all_results = pd.concat(all_results, ignore_index=True)


# Plotting the Cohen's d values
fig, ax = plt.subplots(figsize=(10, 6))

# Default color for the bars
default_color = "darkblue"

bars = ax.barh(
    all_results["Group1"] + " vs. " + all_results["Group2"],
    all_results["Cohen's d"],
    color=default_color,
)

# Set the x-axis starting point at -0.2
ax.set_xlim(
    left=-0.1,
)
# Set the x-axis ending point at 0.9
ax.set_xlim(
    right=0.9,
)

# Draw reference lines for small, medium, and large effect sizes
ax.axvline(x=0.2, color="gray", linestyle="--", label="Small effect size")
ax.axvline(x=0.5, color="gray", linestyle="-.", label="Medium effect size")
ax.axvline(x=0.8, color="gray", linestyle="-", label="Large effect size")

# Label the bars with Cohen's d values
for bar in bars:
    width = bar.get_width()
    label_x_pos = bar.get_width() if bar.get_width() >= 0 else bar.get_width() - 0.05
    ax.text(
        label_x_pos, bar.get_y() + bar.get_height() / 2, f"{width:.2f}", va="center"
    )

ax.set_xlabel("Effect Size (Cohen's d)")
ax.set_title("")
ax.legend()
plt.tight_layout()
plt.show()


# Function to plot regression coefficients
def plot_regression_coefs(model):
    coefs = model.params
    conf = model.conf_int()
    conf["coef"] = coefs
    conf.columns = ["Lower CI", "Upper CI", "Coefficient"]
    conf.reset_index(inplace=True)
    conf.rename(columns={"index": "Variable"}, inplace=True)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Corrected error bar specification
    # Compute the lower and upper error bars
    conf["lower_error"] = conf["Coefficient"] - conf["Lower CI"]
    conf["upper_error"] = conf["Upper CI"] - conf["Coefficient"]
    errors = np.array([conf["lower_error"], conf["upper_error"]])

    ax.errorbar(
        conf["Coefficient"],
        conf["Variable"],
        xerr=errors,
        fmt="o",
        color="blue",
        label="Coefficient (95% CI)",
    )
    ax.axvline(x=0, color="grey", linestyle="--")
    # plt.title("Regression Coefficients and Confidence Intervals")
    plt.title("")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Predictors")
    plt.grid(True)
    plt.legend()
    plt.show()

    
# Call the function to plot the coefficients
plot_regression_coefs(model)



# Regression Analysis Results
regression_results = model.summary()
print("\nRegression Analysis Results:")
print(regression_results)
input("Press Enter to continue...")
with open('regression_results.csv', 'w') as fh:
    fh.write(regression_results.as_csv())

