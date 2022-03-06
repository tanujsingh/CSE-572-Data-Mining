#!/usr/bin/env python
# coding: utf-8

# In[143]:


# import pandas
import pandas as pd

# import numpy
import numpy as np


# In[144]:


#import cgm_data and insulin_data
cgm_data = pd.read_csv('CGMData.csv', low_memory=False, usecols=['Date','Time','Sensor Glucose (mg/dL)'])
insulin_data= pd.read_csv('InsulinData.csv', low_memory=False)


# In[145]:


#make date_time_stamp column using date and time column 
cgm_data['date_time_stamp'] = pd.to_datetime(cgm_data['Date'] + ' ' + cgm_data['Time'])
insulin_data['date_time_stamp'] = pd.to_datetime(insulin_data['Date'] + ' ' + insulin_data['Time'])
cgm_data['hour'] = cgm_data["date_time_stamp"].dt.hour


# In[146]:


#classification of data
cgm_data["hyperglycemia"] = cgm_data["Sensor Glucose (mg/dL)"].apply(lambda x: True if x > 180 else False)
cgm_data["hyperglycemia_critical"] = cgm_data["Sensor Glucose (mg/dL)"].apply(lambda x: True if x > 250 else False)
cgm_data["range"] = cgm_data["Sensor Glucose (mg/dL)"].apply(lambda x: True if x >= 70 and x  <= 180 else False)
cgm_data["range_sec"] = cgm_data["Sensor Glucose (mg/dL)"].apply(lambda x: True if x >= 70 and x  <= 150 else False)
cgm_data["hypoglycemia_L1"] = cgm_data["Sensor Glucose (mg/dL)"].apply(lambda x: True if x < 70 else False)
cgm_data["hypoglycemia_L2"] = cgm_data["Sensor Glucose (mg/dL)"].apply(lambda x: True if x < 54 else False)


# In[147]:


#finding auto_mode and using the earlier one
start_auto_mode = insulin_data.loc[insulin_data['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF']
earlier_auto_mode = start_auto_mode['date_time_stamp'].min()


# In[148]:


# night time data segregation in manaul and auto mode 
cgm_nighttime_data = cgm_data.loc[cgm_data['hour'] < 6]
cgm_nighttime_manual_data = cgm_nighttime_data.loc[cgm_nighttime_data['date_time_stamp'] < earlier_auto_mode]
cgm_nighttime_auto_data = cgm_nighttime_data.loc[cgm_nighttime_data['date_time_stamp'] >= earlier_auto_mode]


# In[149]:


#auto mode night time data classification
cgm_nighttime_auto_total = cgm_nighttime_auto_data.resample('D', on='date_time_stamp').agg({ 'hyperglycemia': 'count' })
cgm_nighttime_auto_total.rename(columns={'hyperglycemia': 'count'}, inplace=True)
cgm_nighttime_auto_daily = cgm_nighttime_auto_data.resample('D', on='date_time_stamp').agg({'hyperglycemia': 'sum','hyperglycemia_critical': 'sum','range': 'sum','range_sec': 'sum','hypoglycemia_L1': 'sum','hypoglycemia_L2': 'sum'})
cgm_nighttime_auto_daily = pd.concat([cgm_nighttime_auto_total, cgm_nighttime_auto_daily], axis='columns', sort=False)
cgm_nighttime_auto_daily = cgm_nighttime_auto_daily[cgm_nighttime_auto_daily['count'] > 0.9*72]
cgm_nighttime_auto_daily['hyperglycemia'] = cgm_nighttime_auto_daily['hyperglycemia']*100/(288*1.0)
cgm_nighttime_auto_daily['hyperglycemia_critical'] = cgm_nighttime_auto_daily['hyperglycemia_critical']*100/(288*1.0)
cgm_nighttime_auto_daily['range'] = cgm_nighttime_auto_daily['range']*100/(288*1.0)
cgm_nighttime_auto_daily['range_sec'] = cgm_nighttime_auto_daily['range_sec']*100/(288*1.0)
cgm_nighttime_auto_daily['hypoglycemia_L1'] = cgm_nighttime_auto_daily['hypoglycemia_L1']*100/(288*1.0)
cgm_nighttime_auto_daily['hypoglycemia_L2'] = cgm_nighttime_auto_daily['hypoglycemia_L2']*100/(288*1.0)


# In[150]:


#manual mode night time data classification
cgm_nighttime_manual_total = cgm_nighttime_manual_data.resample('D', on='date_time_stamp').agg({ 'hyperglycemia': 'count' })
cgm_nighttime_manual_total.rename(columns={'hyperglycemia': 'count'}, inplace=True)
cgm_nighttime_manual_daily = cgm_nighttime_manual_data.resample('D', on='date_time_stamp').agg({'hyperglycemia': 'sum','hyperglycemia_critical': 'sum','range': 'sum','range_sec': 'sum','hypoglycemia_L1': 'sum','hypoglycemia_L2': 'sum'})
cgm_nighttime_manual_daily = pd.concat([cgm_nighttime_manual_total, cgm_nighttime_manual_daily], axis='columns', sort=False)
cgm_nighttime_manual_daily = cgm_nighttime_manual_daily[cgm_nighttime_manual_daily['count'] > 0.9*72]
cgm_nighttime_manual_daily['hyperglycemia'] = cgm_nighttime_manual_daily['hyperglycemia']*100/(288*1.0)
cgm_nighttime_manual_daily['hyperglycemia_critical'] = cgm_nighttime_manual_daily['hyperglycemia_critical']*100/(288*1.0)
cgm_nighttime_manual_daily['range'] = cgm_nighttime_manual_daily['range']*100/(288*1.0)
cgm_nighttime_manual_daily['range_sec'] = cgm_nighttime_manual_daily['range_sec']*100/(288*1.0)
cgm_nighttime_manual_daily['hypoglycemia_L1'] = cgm_nighttime_manual_daily['hypoglycemia_L1']*100/(288*1.0)
cgm_nighttime_manual_daily['hypoglycemia_L2'] = cgm_nighttime_manual_daily['hypoglycemia_L2']*100/(288*1.0)


# In[151]:


#whole day data
cgm_manual_data = cgm_data.loc[cgm_data['date_time_stamp'] < earlier_auto_mode]
cgm_auto_data = cgm_data.loc[cgm_data['date_time_stamp'] >= earlier_auto_mode]


# In[152]:


#auto mode whole day data classification
cgm_auto_total = cgm_auto_data.resample('D', on='date_time_stamp').agg({ 'hyperglycemia': 'count' })
cgm_auto_total.rename(columns={'hyperglycemia': 'count'}, inplace=True)
cgm_auto_daily = cgm_auto_data.resample('D', on='date_time_stamp').agg({'hyperglycemia': 'sum','hyperglycemia_critical': 'sum','range': 'sum','range_sec': 'sum','hypoglycemia_L1': 'sum','hypoglycemia_L2': 'sum'})
cgm_auto_daily = pd.concat([cgm_auto_total, cgm_auto_daily], axis='columns', sort=False)
cgm_auto_daily = cgm_auto_daily[cgm_auto_daily['count'] > 0.9*288]
cgm_auto_daily['hyperglycemia'] = cgm_auto_daily['hyperglycemia']*100/(288*1.0)
cgm_auto_daily['hyperglycemia_critical'] = cgm_auto_daily['hyperglycemia_critical']*100/(288*1.0)
cgm_auto_daily['range'] = cgm_auto_daily['range']*100/(288*1.0)
cgm_auto_daily['range_sec'] = cgm_auto_daily['range_sec']*100/(288*1.0)
cgm_auto_daily['hypoglycemia_L1'] = cgm_auto_daily['hypoglycemia_L1']*100/(288*1.0)
cgm_auto_daily['hypoglycemia_L2'] = cgm_auto_daily['hypoglycemia_L2']*100/(288*1.0)


# In[153]:


#manual mode whole daye data classification
cgm_manual_total = cgm_manual_data.resample('D', on='date_time_stamp').agg({ 'hyperglycemia': 'count' })
cgm_manual_total.rename(columns={'hyperglycemia': 'count'}, inplace=True)
cgm_manual_daily = cgm_manual_data.resample('D', on='date_time_stamp').agg({'hyperglycemia': 'sum','hyperglycemia_critical': 'sum','range': 'sum','range_sec': 'sum','hypoglycemia_L1': 'sum','hypoglycemia_L2': 'sum'})
cgm_manual_daily = pd.concat([cgm_manual_total, cgm_manual_daily], axis='columns', sort=False)
cgm_manual_daily = cgm_manual_daily[cgm_manual_daily['count'] > 0.9*288]
cgm_manual_daily['hyperglycemia'] = cgm_manual_daily['hyperglycemia']*100/(288*1.0)
cgm_manual_daily['hyperglycemia_critical'] = cgm_manual_daily['hyperglycemia_critical']*100/(288*1.0)
cgm_manual_daily['range'] = cgm_manual_daily['range']*100/(288*1.0)
cgm_manual_daily['range_sec'] = cgm_manual_daily['range_sec']*100/(288*1.0)
cgm_manual_daily['hypoglycemia_L1'] = cgm_manual_daily['hypoglycemia_L1']*100/(288*1.0)
cgm_manual_daily['hypoglycemia_L2'] = cgm_manual_daily['hypoglycemia_L2']*100/(288*1.0)


# In[154]:


# daytime data segregation in manaul and auto mode 
cgm_daytime_data = cgm_data.loc[cgm_data['hour'] >= 6]
cgm_daytime_manual_data = cgm_daytime_data.loc[cgm_daytime_data['date_time_stamp'] < earlier_auto_mode]
cgm_daytime_auto_data = cgm_daytime_data.loc[cgm_daytime_data['date_time_stamp'] >= earlier_auto_mode]


# In[155]:


#auto mode day time data classification
cgm_daytime_auto_total = cgm_daytime_auto_data.resample('D', on='date_time_stamp').agg({ 'hyperglycemia': 'count' })
cgm_daytime_auto_total.rename(columns={'hyperglycemia': 'count'}, inplace=True)
cgm_daytime_auto_daily = cgm_daytime_auto_data.resample('D', on='date_time_stamp').agg({'hyperglycemia': 'sum','hyperglycemia_critical': 'sum','range': 'sum','range_sec': 'sum','hypoglycemia_L1': 'sum','hypoglycemia_L2': 'sum'})
cgm_daytime_auto_daily = pd.concat([cgm_daytime_auto_total, cgm_daytime_auto_daily], axis='columns', sort=False)
cgm_daytime_auto_daily = cgm_daytime_auto_daily[cgm_daytime_auto_daily['count'] > 0.9*216]
cgm_daytime_auto_daily['hyperglycemia'] = cgm_daytime_auto_daily['hyperglycemia']*100/(288*1.0)
cgm_daytime_auto_daily['hyperglycemia_critical'] = cgm_daytime_auto_daily['hyperglycemia_critical']*100/(288*1.0)
cgm_daytime_auto_daily['range'] = cgm_daytime_auto_daily['range']*100/(288*1.0)
cgm_daytime_auto_daily['range_sec'] = cgm_daytime_auto_daily['range_sec']*100/(288*1.0)
cgm_daytime_auto_daily['hypoglycemia_L1'] = cgm_daytime_auto_daily['hypoglycemia_L1']*100/(288*1.0)
cgm_daytime_auto_daily['hypoglycemia_L2'] = cgm_daytime_auto_daily['hypoglycemia_L2']*100/(288*1.0)


# In[156]:


#manual mode day time data classification
cgm_daytime_manual_total = cgm_daytime_manual_data.resample('D', on='date_time_stamp').agg({ 'hyperglycemia': 'count' })
cgm_daytime_manual_total.rename(columns={'hyperglycemia': 'count'}, inplace=True)
cgm_daytime_manual_daily = cgm_daytime_manual_data.resample('D', on='date_time_stamp').agg({'hyperglycemia': 'sum','hyperglycemia_critical': 'sum','range': 'sum','range_sec': 'sum','hypoglycemia_L1': 'sum','hypoglycemia_L2': 'sum'})
cgm_daytime_manual_daily = pd.concat([cgm_daytime_manual_total, cgm_daytime_manual_daily], axis='columns', sort=False)
cgm_daytime_manual_daily = cgm_daytime_manual_daily[cgm_daytime_manual_daily['count'] > 0.9*216]
cgm_daytime_manual_daily['hyperglycemia'] = cgm_daytime_manual_daily['hyperglycemia']*100/(288*1.0)
cgm_daytime_manual_daily['hyperglycemia_critical'] = cgm_daytime_manual_daily['hyperglycemia_critical']*100/(288*1.0)
cgm_daytime_manual_daily['range'] = cgm_daytime_manual_daily['range']*100/(288*1.0)
cgm_daytime_manual_daily['range_sec'] = cgm_daytime_manual_daily['range_sec']*100/(288*1.0)
cgm_daytime_manual_daily['hypoglycemia_L1'] = cgm_daytime_manual_daily['hypoglycemia_L1']*100/(288*1.0)
cgm_daytime_manual_daily['hypoglycemia_L2'] = cgm_daytime_manual_daily['hypoglycemia_L2']*100/(288*1.0)


# In[157]:


#creating results dataframe
results_df = pd.DataFrame({"01_overnight_hyperglycemia": [ cgm_nighttime_manual_daily.mean()['hyperglycemia'], cgm_nighttime_auto_daily.mean()['hyperglycemia'] ],
    "02_overnight_hyperglycemia_critical": [ cgm_nighttime_manual_daily.mean()['hyperglycemia_critical'], cgm_nighttime_auto_daily.mean()['hyperglycemia_critical'] ],
    "03_overnight_range": [ cgm_nighttime_manual_daily.mean()['range'], cgm_nighttime_auto_daily.mean()['range'] ],
    "04_overnight_range_sec": [ cgm_nighttime_manual_daily.mean()['range_sec'], cgm_nighttime_manual_daily.mean()['range_sec'] ],
    "05_overnight_hypoglycemia_L1": [ cgm_nighttime_manual_daily.mean()['hypoglycemia_L1'], cgm_nighttime_auto_daily.mean()['hypoglycemia_L1'] ],
    "06_overnight_hypoglycemia_L2": [ cgm_nighttime_manual_daily.mean()['hypoglycemia_L2'], cgm_nighttime_auto_daily.mean()['hypoglycemia_L2'] ],

    "07_daytime_hyperglycemia": [ cgm_daytime_manual_daily.mean()['hyperglycemia'], cgm_daytime_auto_daily.mean()['hyperglycemia'] ],
    "08_daytime_hyperglycemia_critical": [ cgm_daytime_manual_daily.mean()['hyperglycemia_critical'], cgm_daytime_auto_daily.mean()['hyperglycemia_critical'] ],
    "09_daytime_range": [ cgm_daytime_manual_daily.mean()['range'], cgm_daytime_auto_daily.mean()['range'] ],
    "10_daytime_range_sec": [ cgm_daytime_manual_daily.mean()['range_sec'], cgm_daytime_auto_daily.mean()['range_sec'] ],
    "11_daytime_hypoglycemia_L1": [ cgm_daytime_manual_daily.mean()['hypoglycemia_L1'], cgm_daytime_auto_daily.mean()['hypoglycemia_L1'] ],
    "12_daytime_hypoglycemia_L2": [ cgm_daytime_manual_daily.mean()['hypoglycemia_L2'], cgm_daytime_auto_daily.mean()['hypoglycemia_L2'] ],

    "13_wholeday_hyperglycemia": [ cgm_manual_daily.mean()['hyperglycemia'], cgm_auto_daily.mean()['hyperglycemia'] ],
    "14_wholeday_hyperglycemia_critical": [ cgm_manual_daily.mean()['hyperglycemia_critical'], cgm_auto_daily.mean()['hyperglycemia_critical'] ],
    "15_wholeday_range": [ cgm_manual_daily.mean()['range'], cgm_auto_daily.mean()['range'] ],
    "16_wholeday_range_sec": [ cgm_manual_daily.mean()['range_sec'], cgm_auto_daily.mean()['range_sec'] ],
    "17_wholeday_hypoglycemia_L1": [ cgm_manual_daily.mean()['hypoglycemia_L1'], cgm_auto_daily.mean()['hypoglycemia_L1'] ],
    "18_wholeday_hypoglycemia_L2": [ cgm_manual_daily.mean()['hypoglycemia_L2'], cgm_auto_daily.mean()['hypoglycemia_L2'] ], 
                        '19_Dummy': [1.1,1.1]}, index =['manual_mode', 'auto_mode'])


# In[158]:


#getting results in Results.csv file
results_df.to_csv('Results.csv', header=False, index=False)


# In[ ]:




