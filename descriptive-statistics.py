import pandas as pd

# Load the data
df = pd.read_csv(r"\Data\FinalRanking.csv")

print(df)

# Basic descriptive statistics
descriptive_stats = df['Score'].describe()

# Calculating the Interquartile Range (IQR)
Q1 = df['Score'].quantile(0.25)
Q3 = df['Score'].quantile(0.75)
IQR = Q3 - Q1

# Adding IQR to the descriptive stats
descriptive_stats['IQR'] = IQR

# Print the results
print("Descriptive Statistics for Win Rates:\n", descriptive_stats)



# Split the 'Variation' column into separate columns
df[['Colour', 'Imagery', 'Typography']] = df['Variation'].str.split('|', expand=True)

# Calculate descriptive statistics for each category within each independent variable
descriptive_stats_colour = df.groupby('Colour')['Score'].describe()
descriptive_stats_imagery = df.groupby('Imagery')['Score'].describe()
descriptive_stats_typography = df.groupby('Typography')['Score'].describe()

# Add IQR for each independent variable category
descriptive_stats_colour['IQR'] = df.groupby('Colour')['Score'].quantile(0.75) - df.groupby('Colour')['Score'].quantile(0.25)
descriptive_stats_imagery['IQR'] = df.groupby('Imagery')['Score'].quantile(0.75) - df.groupby('Imagery')['Score'].quantile(0.25)
descriptive_stats_typography['IQR'] = df.groupby('Typography')['Score'].quantile(0.75) - df.groupby('Typography')['Score'].quantile(0.25)

# Print the results
print("Descriptive Statistics by Colour:\n", descriptive_stats_colour)
print("\nDescriptive Statistics by Imagery:\n", descriptive_stats_imagery)
print("\nDescriptive Statistics by Typography:\n", descriptive_stats_typography)

# Save the descriptive statistics to CSV files
descriptive_stats_colour.to_csv('descriptive_stats_by_colour.csv')
descriptive_stats_imagery.to_csv('descriptive_stats_by_imagery.csv')
descriptive_stats_typography.to_csv('descriptive_stats_by_typography.csv')



import matplotlib.pyplot as plt

# Plotting boxplots for Colour
plt.figure(figsize=(10, 6))
df.boxplot(column='Score', by='Colour')
plt.title('')
plt.suptitle('')
plt.xlabel('')
plt.ylabel('Win Rate')
plt.savefig('boxplot_scores_by_colour.png')
plt.show()

# Plotting boxplots for Imagery
plt.figure(figsize=(10, 6))
df.boxplot(column='Score', by='Imagery')
plt.title('')
plt.suptitle('')
plt.xlabel('')
plt.ylabel('Win Rate')
plt.savefig('boxplot_scores_by_imagery.png')
plt.show()

# Plotting boxplots for Typography
plt.figure(figsize=(10, 6))
df.boxplot(column='Score', by='Typography')
plt.title('')
plt.suptitle('')
plt.xlabel('')
plt.ylabel('Win Rate')
plt.savefig('boxplot_scores_by_typography.png')
plt.show()

