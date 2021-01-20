import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(file):
    # Load the excel and extract one of the sheets
    xls = pd.ExcelFile(file)
    df = pd.read_excel(xls, 'Inputs')

    # Shuffle the dataset
    df = df.sample(frac=1)

    # Converte a str column to int
    df.loc[df['Status'] == 'Subcooled', 'Status'] = 0
    df.loc[df['Status'] == 'Superheated', 'Status'] = 1

    # Col 0 is ID, Col 1-15 are features
    df_x = df.iloc[:, 1:16]
    df_x['Status'] = df_x['Status'].astype('float32')

    # Cols 16-42 are 27 sensors with varying position on obstacle
    df_y = df.iloc[:, 16:]
    cols = [a for a in list(df_y.columns.values)]

    XY = []
    for i in range(df_y.shape[0]):
        for j in range(len(cols)):
            x = df_x.iloc[i, :].tolist()
            # add one feature to identify which side of the obstackle the sensor is on
            sensor_position = int(cols[j])
            if sensor_position <= 9:
                x.append(1)
            elif sensor_position <= 18:
                x.append(2)
            elif sensor_position <= 21:
                x.append(3)
            elif sensor_position <= 24:
                x.append(4)
            else:
                x.append(5)
            x.append(sensor_position)  # add label of col as feature "Position ID"
            x.append(df_y.iloc[i, j])  # add target values
            XY.append(x)

    columns = list(df_x.columns.values)
    columns.append('obstacle side')
    columns.append('Position ID')
    columns.append('target')

    data = pd.DataFrame(XY, columns=columns)
    data = data[(data.target > 1e-3)]  # get rid of data points with very small target
    missing_values = data.isnull().values.any()
    print(data.columns[data.isnull().any()])
    if missing_values:
        print("===There is Missing value===")

    data.to_excel("data/data_processed.xlsx")

    target = data["target"]
    data.drop("target", axis=1, inplace=True)
    print(data.head(28))
    print(data.columns)

    # dataset split, 70% training 15 validation 15 testing
    n_train = int(data.shape[0] * 0.7)
    n_val = int(data.shape[0] * 0.85)

    train_X = data[:n_train].to_numpy()
    train_y = target[:n_train].to_numpy()
    val_X = data[n_train:n_val].to_numpy()
    val_y = target[n_train:n_val].to_numpy()
    test_X = data[n_val:].to_numpy()
    test_y = target[n_val:].to_numpy()

    train_X = train_X.astype(np.float32)
    train_y = train_y.astype(np.float32)
    val_X = val_X.astype(np.float32)
    val_y = val_y.astype(np.float32)
    test_X = test_X.astype(np.float32)
    test_y = test_y.astype(np.float32)

    # Data preprocessing
    scaler = StandardScaler().fit(train_X)
    mean_X = scaler.mean_
    std_X = scaler.scale_

    # Normalization moved to main.py
    # train_X = scaler.transform(train_X)
    # val_X = scaler.transform(val_X)
    # test_X = scaler.transform(test_X)
    # real_test_X = scaler.transform(real_test_X)

    print(train_X.shape, val_X.shape, test_X.shape)
    return train_X, train_y, val_X, val_y, test_X, test_y, mean_X, std_X


if __name__ == '__main__':
    train_X, train_y, val_X, val_y, test_X, test_y, mean, std = load_data(
        'data/butane_propane_N=8100_D=16.xlsx')
    np.savez('data/BLEVE_Obstacle_Butane_Propane', train_X=train_X, train_y=train_y,
             val_X=val_X, val_y=val_y, test_X=test_X, test_y=test_y,
             mean=mean, std=std)