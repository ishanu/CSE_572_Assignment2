# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle_compat
import pickle

pickle_compat.patch()
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler


def getMeal_Nomealtimes(newTimes, timediff):
    mealTimes = []
    newTimes1 = newTimes[0:len(newTimes) - 1]
    newTimes2 = newTimes[1:len(newTimes)]
    diff = list(np.array(newTimes1) - np.array(newTimes2))
    reqValues = list(zip(newTimes1, newTimes2, diff))
    for j in reqValues:
        if j[2] < timediff:
            mealTimes.append(j[0])
    return mealTimes


def getMeal_NomealData(mealTimes, startTime, endTime, isMealData, new_glucoseData):
    newMealDataRows = []

    for newTime in mealTimes:
        meal_index_start = new_glucoseData[new_glucoseData['datetime'].between(newTime + pd.DateOffset(hours=startTime),
                                                                               newTime + pd.DateOffset(hours=endTime))]
        if meal_index_start.shape[0] < 24:
            continue
        glucoseValues = meal_index_start['Sensor Glucose (mg/dL)'].to_numpy()
        mean = meal_index_start['Sensor Glucose (mg/dL)'].mean()
        if isMealData:
            missing_values_count = 30 - len(glucoseValues)
            if missing_values_count > 0:
                for i in range(missing_values_count):
                    glucoseValues = np.append(glucoseValues, mean)
            newMealDataRows.append(glucoseValues[0:30])
        else:
            newMealDataRows.append(glucoseValues[0:24])
    return pd.DataFrame(data=newMealDataRows)


def dataOperation(insulinData, glucoseData):
    insulinData = insulinData[::-1]
    glucoseData = glucoseData[::-1]
    glucoseData['Sensor Glucose (mg/dL)'] = glucoseData['Sensor Glucose (mg/dL)'].interpolate(method='linear',
                                                                                              limit_direction='both')
    # glucoseData.dropna(subset=['Sensor Glucose (mg/dL)'], inplace=True)
    insulinData['datetime'] = pd.to_datetime(insulinData["Date"].astype(str) + " " + insulinData["Time"].astype(str))
    glucoseData['datetime'] = pd.to_datetime(glucoseData["Date"].astype(str) + " " + glucoseData["Time"].astype(str))

    new_insulinData = insulinData[['datetime', 'BWZ Carb Input (grams)']]
    new_glucoseData = glucoseData[['datetime', 'Sensor Glucose (mg/dL)']]

    new_insulinData = new_insulinData[
        (new_insulinData['BWZ Carb Input (grams)'].notna()) & (new_insulinData['BWZ Carb Input (grams)'] > 0)]

    newTimes = list(new_insulinData['datetime'])

    mealTimes = getMeal_Nomealtimes(newTimes, pd.Timedelta('0 days 120 min'))
    nomealTimes = getMeal_Nomealtimes(newTimes, pd.Timedelta('0 days 240 min'))

    mealData = getMeal_NomealData(mealTimes, -0.5, 2, True, new_glucoseData)
    noMealData = getMeal_NomealData(nomealTimes, 2, 4, False, new_glucoseData)

    mealDataFeatures = glucoseFeatures(mealData)
    noMealDataFeatures = glucoseFeatures(noMealData)

    stdScaler = StandardScaler()
    meal_std = stdScaler.fit_transform(mealDataFeatures)
    noMeal_std = stdScaler.fit_transform(noMealDataFeatures)

    pca = PCA(n_components=5)
    pca.fit(meal_std)

    meal_pca = pd.DataFrame(pca.fit_transform(meal_std))
    noMeal_pca = pd.DataFrame(pca.fit_transform(noMeal_std))

    meal_pca['class'] = 1
    noMeal_pca['class'] = 0

    data = meal_pca.append(noMeal_pca)
    data.index = [i for i in range(data.shape[0])]
    return data


def fn_zero_crossings(row, xAxis):
    slopes = [
        0]
    zero_cross = list()
    zero_crossing_rate = 0
    X = [i for i in range(xAxis)][::-1]
    Y = row[::-1]
    for index in range(0, len(X) - 1):
        slopes.append((Y[(index + 1)] - Y[index]) / (X[(index + 1)] - X[index]))

    for index in range(0, len(slopes) - 1):
        if slopes[index] * slopes[(index + 1)] < 0:
            zero_cross.append([slopes[(index + 1)] - slopes[index], X[(index + 1)]])

    zero_crossing_rate = np.sum(
        [np.abs(np.sign(slopes[(i + 1)]) - np.sign(slopes[i])) for i in range(0, len(slopes) - 1)]) / (2 * len(slopes))
    if len(zero_cross) > 0:
        return [max(zero_cross)[0], zero_crossing_rate]
    else:
        return [
            0, 0]


def absoluteValueMean(param):
    meanValue = 0
    for p in range(0, len(param) - 1):
        meanValue = meanValue + np.abs(param[(p + 1)] - param[p])
    return meanValue / len(param)


def glucoseEntropy(param):
    paramLen = len(param)
    entropy = 0
    if paramLen <= 1:
        return 0
    else:
        value, count = np.unique(param, return_counts=True)
        ratio = count / paramLen
        nonZero_ratio = np.count_nonzero(ratio)
        if nonZero_ratio <= 1:
            return 0
        for i in ratio:
            entropy -= i * np.log2(i)
        return entropy


def rootMeanSquare(param):
    rootMeanSquare = 0
    for p in range(0, len(param) - 1):
        rootMeanSquare = rootMeanSquare + np.square(param[p])
    return np.sqrt(rootMeanSquare / len(param))


def fastFourier(param):
    fastFourier = fft(param)
    paramLen = len(param)
    t = 2 / 300
    amplitude = []
    frequency = np.linspace(0, paramLen * t, paramLen)
    for amp in fastFourier:
        amplitude.append(np.abs(amp))
    sortedAmplitude = amplitude
    sortedAmplitude = sorted(sortedAmplitude)
    max_amplitude = sortedAmplitude[(-2)]
    max_frequency = frequency.tolist()[amplitude.index(max_amplitude)]
    return [max_amplitude, max_frequency]


def glucoseFeatures(meal_Nomeal_data):
    glucoseFeatures = pd.DataFrame()
    for i in range(0, meal_Nomeal_data.shape[0]):
        param = meal_Nomeal_data.iloc[i, :].tolist()
        glucoseFeatures = glucoseFeatures.append({
            'Minimum Value': min(param),
            'Maximum Value': max(param),
            'Mean of Absolute Values1': absoluteValueMean(param[:13]),
            'Mean of Absolute Values2': absoluteValueMean(param[13:]),
            'Root Mean Square': rootMeanSquare(param),
            'Entropy': rootMeanSquare(param),
            'Max FFT Amplitude1': fastFourier(param[:13])[0],
            'Max FFT Frequency1': fastFourier(param[:13])[1],
            'Max FFT Amplitude2': fastFourier(param[13:])[0],
            'Max FFT Frequency2': fastFourier(param[13:])[1]},
            ignore_index=True)
    return glucoseFeatures


if __name__ == '__main__':
    insulinDf = pd.read_csv("InsulinData.csv", low_memory=False)
    glucoseDF = pd.read_csv("CGMData.csv", low_memory=False)
    data = dataOperation(insulinDf, glucoseDF)
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    model = SVC(kernel='linear', C=1, gamma=0.1)
    kfold = KFold(5, True, 1)
    for tr, tst in kfold.split(X, Y):
        X_train, X_test = X.iloc[tr], X.iloc[tst]
        Y_train, Y_test = Y.iloc[tr], Y.iloc[tst]

        model.fit(X_train, Y_train)

    with open('model.pkl', 'wb') as (file):
        pickle.dump(model, file)
