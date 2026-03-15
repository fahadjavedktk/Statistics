import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=13)

data = pd.read_csv("marks_data.csv")

print(data.head())
print(data.columns)

print(data.shape)
print(data.describe())
print(data.info())

print(data.isnull().sum())
print(data.duplicated().sum())

data = data.rename(columns={"S,NO": "SNO"})
data = data.drop(["SNO"], axis=1)

numeric_cols = data.select_dtypes(include=['int64','float64']).columns
non_numeric_cols = data.select_dtypes(include=['object']).columns

print(data[numeric_cols].head())
print(data[non_numeric_cols].head())

cols_to_keep = ['GENDER', 'QUIZ', 'ASSIGN', 'MID', 'FINAL', 'TOTAL', 'LG', 'GP']

data_1 = data[cols_to_keep]

corr_data = data_1.corr(numeric_only=True)

print(data_1['TOTAL'].describe())

plt.figure(figsize=(8,5))
sns.histplot(data_1['TOTAL'], bins=20, color="red", kde=True)
plt.xlabel("TOTAL")
plt.ylabel("Number of Students")
plt.title("Distribution of Score")
plt.savefig("exam_score_distribution.png")
plt.show()

mean_score = data_1["TOTAL"].mean()
std_score = data_1["TOTAL"].std()

plt.figure(figsize=(8,5))
sns.histplot(data_1["TOTAL"], bins=20, color="#EDFFF0")
plt.axvline(mean_score, color="red", linestyle="--", label="Mean")
plt.axvline(mean_score + std_score, color="green", linestyle="--", label="+1 Std Dev")
plt.axvline(mean_score - std_score, color="green", linestyle="--", label="-1 Std Dev")
plt.legend()
plt.title("Exam Score Distribution with Mean and Standard Deviation")
plt.savefig("exam_score_mean_std.png")
plt.show()

gender_counts = data_1["GENDER"].value_counts()

plt.figure(figsize=(5,5))
plt.pie(gender_counts, labels=["M","F"], autopct="%1.1f%%", startangle=90)
plt.title("Gender Distribution of Students")
plt.savefig("gender_distribution_pie.png")
plt.show()