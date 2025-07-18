# Data-Science-Practice
Repo to practice data science topics

Encoders

1. One-Hot Encoding

    Uses sklearn.preprocessing.OneHotEncoder

    Creates binary columns for each unique category.

    Best for: Low cardinality, nominal (unordered) data.

2. Label Encoding

    Uses sklearn.preprocessing.LabelEncoder

    Converts categories to unique integers.

    Best for: Ordinal data or embeddings; not suitable for tree-based models on nominal data.

3. Frequency / Count Encoding

    Replaces categories with their frequency/count in the dataset.

    Simple and preserves information about category prevalence.

    Can work well for tree-based models.

4. Target / Mean Encoding

    Replaces categories with the mean of the target variable grouped by category.

    Useful in supervised settings (regression/classification).

    Can cause target leakage if not handled properly (e.g., without cross-validation or smoothing).

5. Binary Encoding

    Combines hashing and ordinal encoding; converts category indices to binary and spreads across multiple columns.

    Useful for high-cardinality categorical data.

    Implemented using category_encoders.BinaryEncoder.

6. Hash Encoding

    Hashes category names into a fixed number of columns.

    Does not require knowing all categories in advance.

    Useful when working with very high-cardinality data or unseen categories.

7. Embeddings (Deep Learning)

    Converts label-encoded categories into dense, trainable embeddings using torch.nn.Embedding.

    Suitable for deep learning models and handling high-cardinality features in a compact way.

    Example: Category → Label → Embedding vector

Example Usage

Each encoder is implemented as a function, taking a pandas.DataFrame and column name(s), and returning a transformed DataFrame.
