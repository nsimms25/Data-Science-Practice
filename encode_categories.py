import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from scipy.sparse import issparse

import category_encoders as ce

from gen_data import generate_fake_data

#One-hot Encoder
def one_hot_encoder(dataframe : pd.DataFrame):
    encoder = OneHotEncoder(sparse_output=False)
    encoded_array = encoder.fit_transform(dataframe[['Color']])

    transformed_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Color']))

    return transformed_df

def sparse_vs_dense_df(data):
    sparse_encoder = OneHotEncoder(sparse_output=True)
    dense_encoder = OneHotEncoder(sparse_output=False)

    sparse_array = sparse_encoder.fit_transform(data)
    dense_array = dense_encoder.fit_transform(data)

    if issparse(sparse_array):
        sparse_array = sparse_array.toarray() # type: ignore
    
    sparse_df = pd.DataFrame(
        sparse_array,
        columns=sparse_encoder.get_feature_names_out()
    )
    dense_df = pd.DataFrame(
        dense_array,
        columns=dense_encoder.get_feature_names_out()
    )

    return sparse_df, dense_df

#Label Encoding
def label_encoder(dataframe: pd.DataFrame):
    encoder = LabelEncoder()
    encoded_array = encoder.fit_transform(dataframe[["Color"]])
    encoded_array = np.array(encoded_array).reshape(-1, 1)

    transformed_df = pd.DataFrame(encoded_array, columns=['Color_LabelEncoded'])

    return transformed_df

#Frequency / Count Encoding
def count_encoder(dataframe: pd.DataFrame, column: str):
    count = dataframe[column].value_counts().to_dict()
    dataframe[f"count_{column}"] = dataframe[column].map(count)

    return dataframe

#Target / Mean Encoding
def mean_encoder(dataframe: pd.DataFrame, column: str, target: str):
    mean = dataframe.groupby(column)[target].mean().to_dict()
    dataframe[f"{column}_MeanEncode"] = dataframe[column].map(mean)

    return dataframe

#Smoothed Mean Encoding
def smooth_mean_encoder(dataframe: pd.DataFrame, column: str, target: str, smoothing=10):
    global_mean = dataframe[target].mean()

    aggregate = dataframe.groupby(column)[target].aggregate(["mean", "count"])
    means = aggregate['mean']
    counts = aggregate['count']

    smooth = (counts * means + smoothing * global_mean) / (counts + smoothing)

    dataframe[f"{column}_SmoothMeanEncode"] = dataframe[column].map(smooth)

    return dataframe

#Binary Encoding
def binary_encoder(dataframe: pd.DataFrame, column: str):
    categories = dataframe[column].astype("category")
    category_codes = categories.cat.codes

    max_bits = int(np.floor(np.log2(category_codes.max())) + 1)
    binary_cols = (
        category_codes
        .apply(lambda x: format(x, f'0{max_bits}b'))
        .apply(lambda x: pd.Series(list(x)))
        .astype(int)
    )

    binary_cols.columns = [f"{column}_bin_{i}" for i in range(binary_cols.shape[1])]

    return pd.concat([dataframe.reset_index(drop=True), binary_cols], axis=1).drop(columns=[column])

#Hash Encoding
def hash_encoder(dataframe: pd.DataFrame, column: str, n_components: int= 6):
    encoder = ce.HashingEncoder(cols=[column], n_components=n_components, return_df=True)
    encoded_data = encoder.fit_transform(dataframe)

    #This is the avoid a pylance error, truly ensures a DataFrame is returned. 
    if not isinstance(encoded_data, pd.DataFrame):
        encoded_data = pd.DataFrame(encoded_data)

    return encoded_data

#Embeddings (Deep Learning)


if __name__ == "__main__":
    fake_data = generate_fake_data(sample_size=30)
    print(fake_data.head())

    encoded_data = one_hot_encoder(fake_data)
    #Uncomment to show the fake vs encoded.
    #print(fake_data.head())
    #print(encoded_data.head())

    array = fake_data[["Color"]]
    sparse_df, dense_df = sparse_vs_dense_df(array)
    #There are no differences between sparse and dense when showing. But some conversion is needed in the function.
    #print(sparse_df.head())
    #print(dense_df.head())

    label_array = label_encoder(fake_data[["Color"]])
    #Uncomment to show the fake vs encoded.
    #print(fake_data.head())
    #print(label_array.head())

    higher_ord_data = fake_data.copy()
    count_enc_data = count_encoder(dataframe=higher_ord_data, column="City")
    #Uncomment to show the count of City within the DataFrame.
    #print(fake_data.head())
    #print(count_enc_data.head())

    mean_enc_data = mean_encoder(fake_data, "Color", "Purchased")
    #Uncomment to show the mean of purchased per color.
    #print(mean_enc_data.head())

    smooth_mean_enc_data = smooth_mean_encoder(fake_data, 'Color', 'Purchased', smoothing=10)
    #Uncomment to show the smoothed result of the purchase per color.
    #print(fake_data.head())

    bin_data = binary_encoder(fake_data, column="City")
    #Uncomment to show the binary encoder result.
    #print(fake_data.head())
    #print(bin_data.head())

    hash_encoded = hash_encoder(fake_data, column="City")
    #Uncomment to show the hash encoder result.
    #print(fake_data.head())
    #print(hash_encoded.head())
    



