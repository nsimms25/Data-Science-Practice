import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from scipy.sparse import issparse

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

#Binary Encoding

#Hash Encoding

#Embeddings (Deep Learning)


if __name__ == "__main__":
    fake_data = generate_fake_data(sample_size=30)
    print(fake_data)

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

    higher_ord_data = fake_data
    count_enc_data = count_encoder(dataframe=higher_ord_data, column="City")
    #Uncomment to show the count of City within the DataFrame.
    #print(count_enc_data.head())
    #print(fake_data.head())

    