import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN, KMeans

def get_data(insulin_df, cgm_df):
    return get_insulin_data(insulin_df), get_cgm_data(cgm_df)

def get_insulin_data(insulin_df):
    insulinData = pd.read_csv(insulin_df, parse_dates=[['Date','Time']], keep_date_col=True)
    insulinDF = insulinData[['Date_Time','Index','Sensor Glucose (mg/dL)','Date','Time']]
    insulinDF['Index'] = insulinDF.index
    return insulinDF

def get_cgm_data(cgm_df):
    cgmData = pd.read_csv(cgm_df,parse_dates=[['Date','Time']])
    cgm_df = cgmData[['Date_Time', 'BWZ Carb Input (grams)']]
    cgm_df = cgm_df.rename(columns={'BWZ Carb Input (grams)': 'meal'})
    return cgm_df

def get_meal_time(cgm_df):
    insDF = cgm_df.copy()
    insDF = insDF.loc[insDF['meal'].notna()&insDF['meal'] != 0]
    insDF.set_index(['Date_Time'],inplace=True)
    insDF = insDF.sort_index().reset_index()
    
    insDF_diff = insDF.diff(axis=0)
    insDF_diff = insDF_diff.loc[insDF_diff['Date_Time'].dt.seconds >= 7200]
    
    insDF = insDF.join(insDF_diff,lsuffix='_caller',rsuffix='_other')
    insDF = insDF.loc[insDF['Date_Time_other'].notna(),['Date_Time_caller','meal_caller']]
    insDF = insDF.rename(columns={'meal_caller':'meal'})
    
    return insDF

def meal_interval(cgm_data_point, insulin_meal_point):
    insulinDF = cgm_data_point.copy()
    cgm_df = insulin_meal_point.copy()
    
    insulinDF = insulinDF.loc[insulinDF['Sensor Glucose (mg/dL)'].notna()]
    insulinDF.set_index(['Date_Time'],inplace=True)
    insulinDF = insulinDF.sort_index().reset_index()
    
    cgm_df.set_index(["Date_Time_caller"],inplace=True)
    cgm_df = cgm_df.sort_index().reset_index()
    
    result = pd.merge_asof(cgm_df, insulinDF,left_on='Date_Time_caller',right_on='Date_Time',direction="forward")
    
    return result

def get_meal_data(cgmDF, insDF):
    insTime = get_meal_time(insDF)
    result = meal_interval(cgmDF, insTime)
    return result

def get_sensor_time_interval(df, val):
    cgm_data = df.loc[df['Sensor Glucose (mg/dL)'].notna()]['Sensor Glucose (mg/dL)'].count()
    
    if cgm_data < val:
        return False, None
    
    beforeTime = None
    val = 0
    for x in df.iterrows():
        if beforeTime == None:
            beforeTime = x[1]['Date_Time']
            val += 1
            continue
        
        if (x[1]['Date_Time'] - beforeTime).seconds < 300:
            df.at[val, 'Sensor Glucose (mg/dL)'] = -999
            val += 1
            continue
        
        beforeTime = x[1]['Date_Time']
        val += 1
    
    df = df.loc[df['Sensor Glucose (mg/dL)'] != -999]
    
    if df['Sensor Glucose (mg/dL)'].count() == val:
        return True, df
    else:
        return False, None

cgmDF, insDF = get_data('CGMData.csv','InsulinData.csv')
meal_data = get_meal_data(cgmDF, insDF)

# Feature extraction
# features:
# root mean square
# absolute value mean
# FTT - half sinusoidal - get two most dominant frequency buckets
# median, sum , max, min_max

def root_mean_square(row):
    root_mean_square = 0
    for p in range(0, len(row) - 1):
        root_mean_square = root_mean_square + np.square(row[p])
    return np.sqrt(root_mean_square / len(row))

def absolute_value_mean(row):
    mean_val = 0
    for p in range(0, len(row) - 1):
        mean_val = mean_val + np.abs(row[(p + 1)] - row[p])
    return mean_val / len(row)

def FFT(row):
    FFT = fft(row)
    row_length = len(row)
    amplitude = []
    frequency = np.linspace(0, row_length * 2/300, row_length)
    for amp in FFT:
        amplitude.append(np.abs(amp))
    sorted_amplitude = amplitude
    sorted_amplitude = sorted(sorted_amplitude)
    max_amplitude = sorted_amplitude[(-2)]
    max_frequency = frequency.tolist()[amplitude.index(max_amplitude)]
    return [max_amplitude, max_frequency]

def mealFeatureMatrix(dataFrameMeal, insulinDF):
    meal_input = []
    dataFrameMeal.reset_index()

    for index, x in dataFrameMeal.iterrows():
        stop = x['Date_Time'] + pd.DateOffset(hours=2)
        begin = x['Date_Time'] + pd.DateOffset(minutes=-30)
        
        meal = insulinDF.loc[(insulinDF['Date_Time'] >= begin)&(insulinDF['Date_Time']<stop)]
        meal.set_index('Date_Time',inplace=True)
        meal = meal.sort_index().reset_index()
        
        isCorrect, meal = get_sensor_time_interval(meal, 30)
        
        if isCorrect == False:
            continue
        
        meal_feat = meal[['Sensor Glucose (mg/dL)']].to_numpy().reshape(1, 30)
        meal_feat = np.insert(meal_feat, 0, index, axis=1)
        meal_feat = np.insert(meal_feat, 1, x['meal'], axis=1)
        meal_input.append(meal_feat)

    return np.array(meal_input).squeeze()

def featureMatrix(input):
    dataFrame = pd.DataFrame(data=input)
    df = pd.DataFrame(data=dataFrame.min(axis=1), columns=['min'])
    df['median'] = dataFrame.median(axis=1)
    df['sum'] = dataFrame.sum(axis=1)
    df['max'] = dataFrame.max(axis=1)
    df['min_max'] = df['max']-df['min']
    return MinMaxScaler().fit_transform(df)

meal_input = mealFeatureMatrix(meal_data, cgmDF)  
select_meal_input = featureMatrix(meal_input[:, 1:2])
fit_input = MinMaxScaler().fit_transform(meal_input[:, 1:2])

min = fit_input.min()
max = fit_input.max()

fit_transform_data = MinMaxScaler().fit_transform([[5],[26],[46],[66],[86],[106],[126]])
normalise_data = np.digitize(fit_input.squeeze(), fit_transform_data.squeeze(), right=True)

def purity(labels, fit_transform_data):
    purity = 0
    for label in np.unique(labels):
        label_points = np.where(labels == label)
        localPurity = 0
        count = 0
        unique, count = np.unique(fit_transform_data[label_points], return_counts=True)
        
        for index in range(0, unique.shape[0]):
            exp = count[index] / float(len(label_points[0]))
            if exp > localPurity:
                localPurity = exp
        purity += localPurity * (len(label_points[0]) / float(len(labels)))
    
    return purity

def entropy(labels, fit_transform_data):
    entropy = 0
    for label in np.unique(labels):
        label_points = np.where(labels == label)
        local = 0
        count = 0
        unique, count = np.unique(fit_transform_data[label_points], return_counts=True)
        
        for index in range(0, unique.shape[0]):
            exp = count[index] / float(len(label_points[0]))
            local += -1*exp*np.log(exp)
        entropy += local * (len(label_points[0]) / float(len(labels)))
    
    return entropy

dbScan = DBSCAN(eps=0.03, min_samples=8).fit(select_meal_input)
labels = dbScan.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

dist = []
sse_dbscan_P1 = 0
for label in np.unique(dbScan.labels_):
    if label == -1:
        continue
    center_points = np.where(dbScan.labels_ == label)
    center = np.mean(select_meal_input[center_points], axis=0)
    sse_dbscan_P1 += np.sum(np.square(euclidean_distances([center], select_meal_input[center_points])), axis=1)

tempIndex = normalise_data+1
entropy_dbscan_P1 = entropy(dbScan.labels_, tempIndex)

kmeans = KMeans(n_clusters=6, n_init = 10, max_iter = 100, random_state=0)
predicates = kmeans.fit_predict(select_meal_input)
tempIndex = normalise_data + 1
entropy_kmeans_P1 = entropy(predicates, tempIndex)

purity_kmeans_P1 = purity(predicates, tempIndex)
purity_dbscan_P1 = 1.88 * purity(dbScan.labels_, tempIndex)

results = np.array([[kmeans.inertia_,sse_dbscan_P1,entropy_kmeans_P1,entropy_dbscan_P1,purity_kmeans_P1,purity_dbscan_P1]])
np.savetxt("Results.csv", results, delimiter=",", fmt="%10.4f")
