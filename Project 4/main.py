import pandas as pd
import numpy as np
import math
import datetime

# load the data
CGMData_DF = pd.read_csv('CGMData.csv', usecols=["Date", "Time", "Sensor Glucose (mg/dL)"])
InsulinData_DF = pd.read_csv('InsulinData.csv', usecols=["Date","Time","BWZ Carb Input (grams)","BWZ Estimate (U)"])
  
# datetime format
CGMData_DF["DateTime"] = pd.to_datetime(CGMData_DF["Date"].astype(str) + " " + CGMData_DF["Time"].astype(str))
InsulinData_DF["DateTime"] = pd.to_datetime(InsulinData_DF["Date"].astype(str) + " " + InsulinData_DF["Time"].astype(str))
CGMData_DF.drop(columns=["Date", "Time"], inplace=True)
InsulinData_DF.drop(columns=["Date", "Time"], inplace=True)

def set_bin(x, v_min, total_bins):
    a = (x - v_min)/20
    f = math.floor(a)
    if f != total_bins:
        return f
    return f - 1
    
def get_all_meal_DF(InsulinData_DF):
    # get date time of all meals intake
    filter = InsulinData_DF['BWZ Carb Input (grams)'].notnull() & InsulinData_DF['BWZ Carb Input (grams)'] != 0 
    Insulin_all_meal_DF = InsulinData_DF.loc[filter][["DateTime", 'BWZ Carb Input (grams)', 'BWZ Estimate (U)']].sort_values(by="DateTime")
    total_bins = math.ceil((Insulin_all_meal_DF["BWZ Carb Input (grams)"].max() - Insulin_all_meal_DF["BWZ Carb Input (grams)"].min())/20)
    Insulin_all_meal_DF["bin"] = Insulin_all_meal_DF["BWZ Carb Input (grams)"].apply(lambda x: set_bin(x, Insulin_all_meal_DF["BWZ Carb Input (grams)"].min(), total_bins))
    return Insulin_all_meal_DF

InsulinData_all_meal_DF = get_all_meal_DF(InsulinData_DF)

def filter_meals(Insulin_all_meal_DF):
    # check whether another meal is happening or not within 2hrs
    filter = []
    i = 0
    while(i < len(Insulin_all_meal_DF) - 1):
        time_diff = Insulin_all_meal_DF.iloc[i+1]["DateTime"] - Insulin_all_meal_DF.iloc[i]["DateTime"]
        if time_diff.total_seconds() > 2*60*60:
            filter.append(True)
        else:
            filter.append(False)
        i = i+1
    filter.append(True)
    return Insulin_all_meal_DF[filter]

InsulinData_meal_DF = filter_meals(InsulinData_all_meal_DF)

def extract_meal_CGM_data(Insulin_meal_DF, CGMData_DF):
    # extracting meal data
    meal_data_bins = []
    ib_datas = []
    mealData_DF = pd.DataFrame()
    
    i = 0
    while (i < len(Insulin_meal_DF)):
        upper_bound = Insulin_meal_DF.iloc[i]["DateTime"] + datetime.timedelta(seconds=2*60*60)
        lower_bound = Insulin_meal_DF.iloc[i]["DateTime"] - datetime.timedelta(seconds=30*60)
        filter = (CGMData_DF["DateTime"] >= lower_bound) & (CGMData_DF["DateTime"] < upper_bound)
        filtered_CGMData_DF = CGMData_DF[filter]
        ib_data = int(round(Insulin_meal_DF.iloc[i]["BWZ Estimate (U)"]))
        meal_bin = Insulin_meal_DF.iloc[i]["bin"]        
        if len(filtered_CGMData_DF.index) == 30 and filtered_CGMData_DF.isnull().values.any() != True:
            filtered_CGMData_DF = filtered_CGMData_DF.sort_values(by="DateTime").T
            filtered_CGMData_DF.drop('DateTime', inplace=True)
            filtered_CGMData_DF.columns = list(range(1,31))
            filtered_CGMData_DF.reset_index(drop=True, inplace=True)
            mealData_DF = mealData_DF.append(filtered_CGMData_DF, ignore_index=True)
            meal_data_bins.append(meal_bin)
            ib_datas.append(ib_data)
        i = i+1
    return mealData_DF.apply(pd.to_numeric), np.array(ib_datas), np.array(meal_data_bins)

mealData_DF, ibData, meal_data_bins  = extract_meal_CGM_data(InsulinData_meal_DF, CGMData_DF)

# get bins
total_CGM_bins = math.ceil((mealData_DF.max().max() -  mealData_DF.min().min())/20)

bmax = mealData_DF.max(axis=1).apply(lambda x: set_bin(x, mealData_DF.min().min(), total_CGM_bins))
bmeal = mealData_DF[7].apply(lambda x: set_bin(x, mealData_DF.min().min(), total_CGM_bins))

all_rules = {}
all_antecedents = {}
i = 0
while(i < len(bmeal)):
    ant_key = f'{bmax[i]}_{bmeal[i]}'
    if ant_key not in all_antecedents:
        all_antecedents[ant_key] = 1
    else:
        all_antecedents[ant_key] = all_antecedents[ant_key] + 1
    
    key = f'{bmax[i]}_{bmeal[i]}_{ibData[i]}'
    if key not in all_rules:
        all_rules[key] = 1
    else:
        all_rules[key] = all_rules[key] + 1
    i = i+1

all_rules_DF = pd.DataFrame.from_dict(all_rules, orient='index', columns=["frequency"])
all_rules_DF_index = all_rules_DF.index.to_series()

def getAntFromRule(rule):
    splits = rule.split("_")
    rule1 = splits[0]
    rule2 = splits[1]
    return "_".join((rule1, rule2))

all_rules_DF['ant'] = all_rules_DF_index.apply(lambda x: getAntFromRule(x))

def getItemsetStr(rule):
    splits = rule.split("_")
    rule1 = splits[0]
    rule2 = splits[1]
    rule3 = splits[2]
    return f'{{{rule1}, {rule2}, {rule3}}}'

all_rules_DF['itemset'] = all_rules_DF_index.apply(lambda x: getItemsetStr(x))

def getRuleStr(rule):
    splits = rule.split("_")
    rule1 = splits[0]
    rule2 = splits[1]
    rule3 = splits[2]
    return f'{{{rule1}, {rule2}}}: {{{rule3}}}'

all_rules_DF['rule'] = all_rules_DF_index.apply(lambda x: getRuleStr(x))
all_rules_DF['confidence'] = all_rules_DF.apply(lambda x: float(x["frequency"]/all_antecedents[x["ant"]]), axis = 1)

all_rules_DF1 = all_rules_DF.loc[:,["itemset", "frequency"]]
all_rules_DF1 = all_rules_DF1.sort_values(by="frequency", ascending=False)
all_rules_DF1.to_csv("./result1.csv", index=False)

all_rules_DF2 = all_rules_DF.loc[:,["rule", "confidence"]]
all_rules_DF2 = all_rules_DF2.sort_values(by="confidence", ascending=False)
all_rules_DF2.to_csv("./result2.csv", index=False)

all_rules_DF3 = all_rules_DF.loc[all_rules_DF["confidence"] < 0.15,["rule", "confidence"]]
all_rules_DF3 = all_rules_DF3.sort_values(by="confidence", ascending=False)
all_rules_DF3.to_csv("./result3.csv", index=False)