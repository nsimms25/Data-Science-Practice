import pandas as pd
import numpy as np

np.random.seed(42)

sample_size = 20

# Low cardinality
colors = ['Red', 'Blue', 'Green']

# Ordinal categorical
education_levels = ['High School', 'Bachelors', 'Masters', 'PhD']

# This is technically high cardinality
cities = [f"City_{i}" for i in range(10)]

ages = np.random.randint(20, 60, size=sample_size)
incomes = np.random.randint(40000, 120000, size=sample_size)

# Target variable (regression or classification)
target = np.random.randint(0, 2, size=sample_size)

df = pd.DataFrame({
    'Color': np.random.choice(colors, size=sample_size),
    'Education': np.random.choice(education_levels, size=sample_size),
    'City': np.random.choice(cities, size=sample_size),
    'Age': ages,
    'Income': incomes,
    'Purchased': target
})

print(df.head())
