import pandas as pd
import numpy as np
import pickle
import pickle_compat
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from scipy.fftpack import fft

pickle_compat.patch()


def pcaAnalysis(mealDataFeatures, noMealDataFeatures):
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


def mealNoMealExtraction(insulinPath, cgmPath):
    df_insulin = pd.read_csv(insulinPath, parse_dates=[['Date', 'Time']], keep_date_col=True)
    df_cgm = pd.read_csv(cgmPath, parse_dates=[['Date', 'Time']], keep_date_col=True)

    # copy the non NaN carb input to another dataframe
    df_insulin_meal = df_insulin[df_insulin[['BWZ Carb Input (grams)']].notnull().all(1)]
    # taking non nan and non zero values only of column Y
    df_insulin_meal = df_insulin_meal[df_insulin_meal['BWZ Carb Input (grams)'] != 0]
    df_insulin_meal.reset_index(inplace=True)
    # get the start time of carb intake from insulin data
    carb_start_time = min(df_insulin_meal['Date_Time'])

    df_insulin_meal['Time_Diff'] = (abs(carb_start_time - df_insulin_meal['Date_Time']))

    time_diff_list = []
    for each in df_insulin_meal['Time_Diff']:
        time_diff_list.append(each.total_seconds() / 60 ** 2)

    df_insulin_meal['Time_Diff'] = pd.DataFrame(time_diff_list, columns=['Time_Diff'])

    max_index = max(df_insulin_meal.index)  # id len(df_insulin_meal)
    j = max_index
    while (j):

        tm = df_insulin_meal.loc[j]['Date_Time']

        tp = df_insulin_meal.loc[j - 1]['Date_Time']

        if ((tp > tm) and ((tp - tm).total_seconds() / 60 ** 2) < 2):
            df_insulin_meal.at[j - 1, 'Meal/No Meal'] = 1
            df_insulin_meal.at[j, 'Meal/No Meal'] = 0


        elif (((tp - tm).total_seconds() / 60 ** 2) == 2):
            df_insulin_meal.at[j - 1, 'Meal/No Meal'] = 1



        else:
            df_insulin_meal.at[j, 'Meal/No Meal'] = 1
            df_insulin_meal.at[j - 1, 'Meal/No Meal'] = 1

        j = j - 1

    j = 1
    tm = df_insulin_meal.loc[j]['Date_Time']

    tp = df_insulin_meal.loc[j - 1]['Date_Time']

    if ((tp > tm) and ((tp - tm).total_seconds() / 60 ** 2) < 2):
        df_insulin_meal.at[j - 1, 'Meal/No Meal'] = 1
        df_insulin_meal.at[j, 'Meal/No Meal'] = 0


    elif (((tp - tm).total_seconds() / 60 ** 2) == 2):
        df_insulin_meal.at[j - 1, 'Meal/No Meal'] = 1


    else:
        df_insulin_meal.at[j, 'Meal/No Meal'] = 1
        df_insulin_meal.at[j - 1, 'Meal/No Meal'] = 1

    df_insulin_meal.set_index('index', inplace=True)
    df_insulin_meal = df_insulin_meal[df_insulin_meal['Meal/No Meal'] == 1]
    df_insulin_meal.reset_index(inplace=True)

    max_index_insulin_meal = max(df_insulin_meal.index)  # id you can use len(df_insulin_meal)
    max_index_cgm = max(df_cgm.index)  # id you can use len(df_insulin_meal)

    i_meal = max_index_insulin_meal

    # df_cgm['Meal/No Meal']=999999
    # marking meal times in cgm data from insulin meal times
    df_cgm.insert(loc=0, column='Meal/No Meal', value=np.nan)
    while (i_meal):
        meal_time = df_insulin_meal.at[i_meal, 'Date_Time']
        #    print(meal_time)
        dict1 = {}
        i_cgm = max_index_cgm
        while i_cgm:

            cgm_time = df_cgm.at[i_cgm, 'Date_Time']
            #        print(cgm_time)
            if cgm_time > meal_time and ((cgm_time - meal_time).total_seconds() <= (2 * 3600)):  # id
                df_cgm.at[i_cgm, 'Meal/No Meal'] = 1
                i_cgm = i_cgm - 1
            #            print(delta)
            elif cgm_time < meal_time:
                i_cgm = i_cgm - 1
            else:
                max_index_cgm = i_cgm
                break

        #    print(dict1)

        ##temp = min(dict1.values())
        ##res = [key for key in dict1 if dict1[key] == temp]
        # df_cgm.at[res, 'Meal/No Meal'] = 1
        i_meal = i_meal - 1

    # meal_start_pt = df_cgm[df_cgm['Meal/No Meal'] == 1].index
    # create meal data with two hours span
    '''for each in meal_start_pt:
        tm = df_cgm.at[each, 'Date_Time']
        k = each - 1
        while (k):
            tm1 = df_cgm.at[k, 'Date_Time']
            if ((tm1 - tm).total_seconds() / 60 ** 2) <= 2:
                df_cgm.at[k, 'Meal/No Meal'] = 1
            else:
                break
            k = k - 1'''

    df_cgm_meal = df_cgm
    # add 30 min data to 2hr meal data:
    idx = 0

    while idx < len(df_cgm):
        while df_cgm.loc[idx]['Meal/No Meal'] != 1:
            idx = idx + 1
        while (idx < len(df_cgm)):

            if (df_cgm_meal.at[idx, 'Meal/No Meal'] == 1):
                tm = df_cgm.at[idx, 'Date_Time']
                idx = idx + 1
                continue
            else:
                tm = df_cgm.at[idx, 'Date_Time']
                tm30 = df_cgm_meal.at[idx, 'Date_Time']
                while ((tm - tm30).total_seconds() <= 30 * 60):
                    df_cgm_meal.at[idx, 'Meal/No Meal'] = 1
                    idx = idx + 1
                    tm30 = df_cgm_meal.at[idx, 'Date_Time']
            break

    '''while idx < len(df_cgm):
        tm = df_cgm.at[idx, 'Date_Time']
        k = idx + 1
        while (k):
            if (df_cgm_meal.at[k, 'Meal/No Meal']==1):
                continue
            else:
                tm30 = df_cgm_meal.at[k, 'Date_Time']
                if ((tm - tm30).total_seconds() / 60 <= 30):
                    df_cgm_meal.at[k, 'Meal/No Meal'] = 1
                else:
                    break
            k = k + 1'''

    # df_cgm_meal = df_cgm[df_cgm['Meal/No Meal']==1]

    # create No meal data with two hours span
    meal_instances = df_cgm[df_cgm['Meal/No Meal'] == 1].index

    # my code
    meal_date_time = df_cgm[df_cgm['Meal/No Meal'] == 1]['Date_Time']
    # j = len(meal_date_time) -1
    k = j = len(df_cgm) - 1

    if df_cgm.loc[j]['Meal/No Meal'] == 1:
        while df_cgm.loc[j]['Meal/No Meal'] == 1:
            j = j - 1
        k = j

        if df_cgm.loc[j]['Meal/No Meal'] != 1:
            while j > 0 and k > 0:
                while df_cgm.loc[j]['Meal/No Meal'] != 1 and j > 0:
                    j = j - 1
                    if j == 0:
                        break
                while k > 0:
                    if (df_cgm.loc[j]['Date_Time'] - df_cgm.loc[k]['Date_Time']).total_seconds() > 2 * 3600:
                        temp = df_cgm.loc[k]['Date_Time']
                        while (df_cgm.loc[k]['Date_Time'] - temp).total_seconds() <= 2 * 3600:
                            df_cgm_meal.at[k, 'Meal/No Meal'] = 0
                            k = k - 1
                    else:
                        while (df_cgm.loc[j]['Meal/No Meal'] == 1 and j > 0):
                            j = j - 1
                            k = j
                        break

    # id
    df_cgm_no_meal = df_cgm[df_cgm['Meal/No Meal'] == 0]
    df_cgm_meal = df_cgm[df_cgm['Meal/No Meal'] == 1]
    # df_cgm_no_meal = pd.read_csv("df_cgm_no_meal.csv")
    # df_cgm_meal = pd.read_csv("df_cgm_meal.csv")

    meal_index = df_cgm_meal.index
    no_meal_index = df_cgm_no_meal.index

    mi = len(meal_index) - 1
    nmi = len(no_meal_index) - 1

    # taking 30 meal data in rows since 2:30 hrs
    k = mi

    cgm_meal = {}
    while (k >= 0):
        cgm = []
        for each in range(0, 30):
            if (k > 0):
                index_start = meal_index[k]
                next_index = meal_index[k - 1]
                if ((pd.to_datetime(df_cgm_meal.at[index_start, 'Date_Time']) - pd.to_datetime(df_cgm_meal.at[
                                                                                                   next_index, 'Date_Time'])).total_seconds() / 60 <= 5):
                    cgm.append(df_cgm_meal.at[index_start, 'Sensor Glucose (mg/dL)'])
                    k = k - 1
                else:
                    cgm.append(df_cgm_meal.at[next_index, 'Sensor Glucose (mg/dL)'])
                    break
        if (len(cgm) == 30):
            cgm_meal.update({k: cgm})
        k = k - 1

        # taking 24 no meal data in rows since 2 hrs
    k = nmi

    no_meal = {}

    while (k >= 0):
        cgm_no_meal = []
        for each in range(0, 24):
            if (k > 0):
                index_start = no_meal_index[k]
                next_index = no_meal_index[k - 1]
                if ((pd.to_datetime(df_cgm_no_meal.at[index_start, 'Date_Time']) - pd.to_datetime(df_cgm_no_meal.at[
                                                                                                      next_index, 'Date_Time'])).total_seconds() / 60 <= 5):
                    cgm_no_meal.append(df_cgm_no_meal.at[index_start, 'Sensor Glucose (mg/dL)'])
                    k = k - 1
                else:
                    cgm_no_meal.append(df_cgm_no_meal.at[next_index, 'Sensor Glucose (mg/dL)'])
                    break
        if (len(cgm_no_meal) == 24):
            no_meal.update({k: cgm_no_meal})
        k = k - 1

        # preparing Nx30 matrix for meal data
    meal_data_matrix = pd.DataFrame(cgm_meal)
    meal_data_matrix = meal_data_matrix.transpose()
    meal_data_matrix = meal_data_matrix.dropna()

    # Px24 Mmatrix for no meal data
    no_meal_data_matrix = pd.DataFrame(no_meal)
    no_meal_data_matrix = no_meal_data_matrix.transpose()
    no_meal_data_matrix = no_meal_data_matrix.dropna()
    mealFeatures = glucoseFeatures(meal_data_matrix)
    noMealFeatures = glucoseFeatures(no_meal_data_matrix)

    data = pcaAnalysis(mealFeatures,noMealFeatures)

    return data

#X_pt1, Y_pt1 = mealNoMealExtraction('InsulinData.csv', 'CGMData.csv')

# X_pt2, Y_pt2 = mealNoMealExtraction('Insulin_patient2.csv', 'CGM_patient2.csv')


from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

# svm.SVC()
# model=svm.SVC(kernel='linear',C=.01,gamma=0.1) #svc for classification: kernel linear performance of C=.1 is best so far
# sv=model.fit(X_pt1,Y_pt1)
# joblib.dump(sv,'DMpt1.pkl')

# model_from_job = joblib.load('DMpt1.pkl')

# Y_pred = model_from_job.predict(X_pt2)

# Y_pred = sv.predict(X_pt2)
# acc_score=accuracy_score(Y_pt2,Y_pred)
# print("SVM KERNEL LINEAR ACCURACY C1 GAMMA1 IS: ",acc_score*100)
# precision_recall_fscore_support(Y_pt2, Y_pred, average='binary',pos_label=0)


from sklearn.naive_bayes import GaussianNB

'''clf = GaussianNB()
model=clf.fit(X_pt1,Y_pt1)
#joblib.dump(model,'DMptG.pkl')
with open('DMptG.pkl', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

svm.SVC()
model = svm.SVC(kernel='linear', C=.01,
                gamma=0.1)  # svc for classification: kernel linear performance of C=.1 is best so far
sv = model.fit(X_pt1, Y_pt1)

with open('model.pkl', 'wb') as handle:
    pickle.dump(sv, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('model.pkl', 'rb') as handle:
    model_from_job = pickle.load(handle)'''
from sklearn.svm import SVC
if __name__=='__main__':

    data = mealNoMealExtraction('InsulinData.csv', 'CGMData.csv')
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


