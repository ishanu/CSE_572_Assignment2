
import pandas as pd
import numpy as np
import pickle
import pickle_compat
pickle_compat.patch()

def meal_NoMeal_Data_Extract(file_path_insulin, file_path_cgm):
    df_insulin = pd.read_csv(file_path_insulin, parse_dates=[['Date', 'Time']],
                             keep_date_col=True)  # read insulin data and copy it to a dataframe
    df_cgm = pd.read_csv(file_path_cgm, parse_dates=[['Date', 'Time']], keep_date_col=True)

    #df_cgm.dropna(subset=['Sensor Glucose (mg/dL)'], inplace=True)

    # copy the non NaN carb input to another dataframe
    df_insulin_meal = df_insulin[df_insulin[['BWZ Carb Input (grams)']].notnull().all(1)]
    # taking non nan and non zero values only of column Y
    df_insulin_meal = df_insulin_meal[df_insulin_meal['BWZ Carb Input (grams)'] != 0]
    df_insulin_meal.reset_index(inplace=True)
    # get the start time of carb intake from insulin data
    carb_start_time = min(df_insulin_meal['Date_Time'])

    df_insulin_meal['Time_Diff'] = (abs(carb_start_time - df_insulin_meal['Date_Time']))
    #
    # time_diff_list=[]
    # for each in df_insulin_meal['Time_Diff']:
    #    time_diff_list.append(each.total_seconds())
    ##store time diff in cgm data frame
    # df_insulin_meal['Time_Diff']=pd.DataFrame(time_diff_list,columns=['Time_Diff'])
    time_diff_list = []
    for each in df_insulin_meal['Time_Diff']:
        time_diff_list.append(each.total_seconds() / 60 ** 2)
    # store time diff in cgm data frame
    df_insulin_meal['Time_Diff'] = pd.DataFrame(time_diff_list, columns=['Time_Diff'])
    ##marking the meal time in insulin data
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
    #  df_cgm=pd.read_excel(file_path_cgm,parse_dates=[['Date', 'Time']],keep_date_col=True)

    max_index_insulin_meal = max(df_insulin_meal.index)  # id you can use len(df_insulin_meal)
    max_index_cgm = max(df_cgm.index)  # id you can use len(df_insulin_meal)

    i_meal = max_index_insulin_meal
    i_cgm = max_index_cgm  # id this line is redundant

    # df_cgm['Meal/No Meal']=999999
    # marking meal times in cgm data from insulin meal times
    df_cgm.insert(loc=0,column='Meal/No Meal', value=np.nan)
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
                i_cgm = i_cgm-1
            else:
                max_index_cgm = i_cgm
                break

        #    print(dict1)

        ##temp = min(dict1.values())
        ##res = [key for key in dict1 if dict1[key] == temp]
        #df_cgm.at[res, 'Meal/No Meal'] = 1
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
        while (idx<len(df_cgm)):

            if (df_cgm_meal.at[idx, 'Meal/No Meal']==1):
                tm = df_cgm.at[idx, 'Date_Time']
                idx = idx + 1
                continue
            else:
                tm = df_cgm.at[idx, 'Date_Time']
                tm30 = df_cgm_meal.at[idx, 'Date_Time']
                while ((tm - tm30).total_seconds() <= 30*60):
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

    #my code
    meal_date_time = df_cgm[df_cgm['Meal/No Meal'] == 1]['Date_Time']
   # j = len(meal_date_time) -1
    k= j = len(df_cgm) - 1


    if df_cgm.loc[j]['Meal/No Meal'] == 1:
        while df_cgm.loc[j]['Meal/No Meal'] == 1:
            j = j - 1
        k = j

        if df_cgm.loc[j]['Meal/No Meal'] != 1:
            while j>0 and k> 0:
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


    #id
    df_cgm_no_meal = df_cgm[df_cgm['Meal/No Meal'] == 0]
    df_cgm_meal = df_cgm[df_cgm['Meal/No Meal'] == 1]
    #df_cgm_no_meal = pd.read_csv("df_cgm_no_meal.csv")
    #df_cgm_meal = pd.read_csv("df_cgm_meal.csv")

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
    meal_data_matrix.to_csv('meal.csv')

    # Px24 Mmatrix for no meal data
    no_meal_data_matrix = pd.DataFrame(no_meal)
    no_meal_data_matrix = no_meal_data_matrix.transpose()
    no_meal_data_matrix = no_meal_data_matrix.dropna()
    no_meal_data_matrix.to_csv('nomeal.csv')
    # feature extraction of meal data:

    ## FEATURE1: TMAX-TM
    '''tmax_tm = []
    for each in meal_data_matrix.idxmax(axis=1) * 5 - 30:
        if each < 0:
            tmax_tm.append(each + 30)
        else:
            tmax_tm.append(each)
    tmax_tm = pd.DataFrame(tmax_tm)'''

    tmax_tm = (meal_data_matrix.idxmax(axis=1) * 5) - 30

    ##FEATURE2: CGM max - CGM min
    cgm_diff = meal_data_matrix.max(axis=1) - meal_data_matrix.min(axis=1)

    ##FEATURE 3 AND 4 Velocity max and time at which velocity is max
    meal_data_v = meal_data_matrix.diff(axis=1)
    v_max = meal_data_v.max(axis=1)
    t_vmax = meal_data_v.idxmax(axis=1) * 5

    ##FEATURE 4: powers
    x_array = meal_data_matrix.values
    f1 = []
    f2 = []
    for each in x_array:
        ps = 2 * np.abs(np.fft.fft(each))
        ls = []
        for p1 in ps:
            ls.append(round(p1, 2))
        ls = set(ls)
        ls = list(ls)
        ls.sort()
        w1 = ls[-2]
        w2 = ls[-3]
        f1.append(w1)
        f2.append(w2)

    dff1 = pd.DataFrame(f1)
    dff2 = pd.DataFrame(f2)

    ##FEATURE 5: Windowed mean and standard deviation
    df_len = len(meal_data_matrix)
    m1 = []
    m2 = []
    m3 = []
    d1 = []
    d2 = []
    d3 = []
    for each in range(0, df_len):
        df_test = meal_data_matrix.iloc[each]
        m1.append(sum(df_test[10:15]) / 5)
        m2.append(sum(df_test[15:20]) / 5)
        m3.append(sum(df_test[20:25]) / 5)
        d1.append(df_test[10:15].std())
        d2.append(df_test[15:20].std())
        d3.append(df_test[20:25].std())

        dfm1 = pd.DataFrame(m1)
        dfm2 = pd.DataFrame(m2)
        dfm3 = pd.DataFrame(m3)

    dfd1 = pd.DataFrame(d1)
    dfd2 = pd.DataFrame(d2)
    dfd3 = pd.DataFrame(d3)

    ##concatenating the features:
    meal_feature_matrix = pd.concat([tmax_tm, cgm_diff, v_max, t_vmax], axis=1, ignore_index=True)
    meal_feature_matrix.reset_index(inplace=True)
    meal_feature_matrix = pd.concat([meal_feature_matrix, dff1, dff2, dfm1, dfm2, dfm3, dfd1, dfd2, dfd3], axis=1)
    #    meal_feature_matrix = pd.concat([meal_feature_matrix,dff1,dff2],axis=1)
    meal_feature_matrix.drop(columns='index', inplace=True)

    meal_feature_matrix.columns = (range(0, 12))
    meal_feature_matrix['Label'] = 1
    #    meal_feature_matrix = meal_feature_matrix.sample(n=275)

    # feature extraction of no meal data:
    ## FEATURE1: TMAX-TM
    tmax_tm = (no_meal_data_matrix.idxmax(axis=1)*5)

    '''tmax_tm = []
    for each in meal_data_matrix.idxmax(axis=1) * 5 - 30:
        if each < 0:
            tmax_tm.append(each + 30)
        else:
            tmax_tm.append(each)
    tmax_tm = pd.DataFrame(tmax_tm)'''

    ##FEATURE2: CGM max - CGM min
    cgm_diff = no_meal_data_matrix.max(axis=1) - no_meal_data_matrix.min(axis=1)

    ##FEATURE 3 AND 4 Velocity max and time at which velocity is max
    meal_data_v = no_meal_data_matrix.diff(axis=1)
    v_max = meal_data_v.max(axis=1)
    t_vmax = meal_data_v.idxmax(axis=1) * 5

    ##FEATURE 4: powers
    x_array = no_meal_data_matrix.values
    f1 = []
    f2 = []
    for each in x_array:
        ps = 2 * np.abs(np.fft.fft(each))
        ls = []
        for p1 in ps:
            ls.append(round(p1, 2))
        ls = set(ls)
        ls = list(ls)
        ls.sort()
        w1 = ls[-2]
        w2 = ls[-3]
        f1.append(w1)
        f2.append(w2)

    dff1 = pd.DataFrame(f1)
    dff2 = pd.DataFrame(f2)

    ##FEATURE 5: Windowed mean and standard deviation
    df_len = len(no_meal_data_matrix)
    m1 = []
    m2 = []
    m3 = []
    d1 = []
    d2 = []
    d3 = []
    for each in range(0, df_len):
        df_test = no_meal_data_matrix.iloc[each]
        m1.append(sum(df_test[10:15]) / 5)
        m2.append(sum(df_test[15:20]) / 5)
        m3.append(sum(df_test[20:25]) / 5)
        d1.append(df_test[10:15].std())
        d2.append(df_test[15:20].std())
        d3.append(df_test[20:25].std())

    dfm1 = pd.DataFrame(m1)
    dfm2 = pd.DataFrame(m2)
    dfm3 = pd.DataFrame(m3)

    dfd1 = pd.DataFrame(d1)
    dfd2 = pd.DataFrame(d2)
    dfd3 = pd.DataFrame(d3)

    ##concatenating the features:
    no_meal_feature_matrix = pd.concat([tmax_tm, cgm_diff, v_max, t_vmax], axis=1, ignore_index=True)
    no_meal_feature_matrix.reset_index(inplace=True)
    no_meal_feature_matrix = pd.concat([no_meal_feature_matrix, dff1, dff2, dfm1, dfm2, dfm3, dfd1, dfd2, dfd3], axis=1)
    #    no_meal_feature_matrix = pd.concat([no_meal_feature_matrix,dff1,dff2],axis=1)
    no_meal_feature_matrix.drop(columns='index', inplace=True)

    no_meal_feature_matrix.columns = (range(0, 12))
    no_meal_feature_matrix['Label'] = 0
    #    no_meal_feature_matrix = no_meal_feature_matrix.sample(n=275)

    pt_data = pd.concat([meal_feature_matrix, no_meal_feature_matrix], ignore_index=True)
    features = (range(0, 12))
    target = ['Label']
    X = pt_data[features]
    Y = pt_data[target]
    return X, Y


file_path_insulin = "0"  # input('Please enter full path of Patient 1 inslin data file: ')
file_path_cgm = "0"  # input('Please enter full path of Patient 1 CGM data file: ')

X_pt1, Y_pt1 = meal_NoMeal_Data_Extract('InsulinData.csv', 'CGMData.csv')

#X_pt2, Y_pt2 = meal_NoMeal_Data_Extract('Insulin_patient2.csv', 'CGM_patient2.csv')



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
'''
svm.SVC()
model = svm.SVC(kernel='linear', C=.01,
                gamma=0.1)  # svc for classification: kernel linear performance of C=.1 is best so far
sv = model.fit(X_pt1, Y_pt1)

with open('DMptG.pkl', 'wb') as handle:
    pickle.dump(sv, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('DMptG.pkl', 'rb') as handle:
    model_from_job = pickle.load(handle)

# model_from_job = pickle.load('DMpt1.pkl')

#Y_pred = model_from_job.predict(X_pt2)

#Y_pred = sv.predict(X_pt2)
#acc_score = accuracy_score(Y_pt2, Y_pred)
#print("SVM KERNEL LINEAR ACCURACY C1 GAMMA1 IS: ", acc_score * 100)
#precision_recall_fscore_support(Y_pt2, Y_pred, average='binary', pos_label=0)'''

#print('done')

