# Databricks notebook source
# MAGIC %pip install pmdarima
# MAGIC %pip install holidays
# MAGIC %pip install prophet
# MAGIC %pip install lightgbm
# MAGIC %pip install pyod  

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pprint import pprint
import os
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import unicodedata
import dateutil.parser
from dateutil import tz
import re
import plotly
from datetime import datetime
import pyspark.pandas as ps
import statsmodels.api as sm
from datetime import timedelta
from pmdarima.arima import auto_arima
from prophet import Prophet
from prophet.diagnostics import cross_validation
import lightgbm as lgb
import holidays
import itertools
from datetime import date
from plotly.offline import plot


# COMMAND ----------

ccu_agents = ['CUSLSH',	'CPFKSUA',	'CPFWONGS',	'CPFROSAN',	'CPFABHMS',	'CPFAAKMB',	'CPFPEIW',	'CPFJCHOW',	'CPFEVIE',	'CPFNAIN',	'CPFAAMKP',	'CPFEDEL',	'CPFAAOYT',	'CPFAAAGS',	'CPFAADTW',	'CUSKH',	'CPFJASLY',	'CPFJYAP',	'RYDDKY',	'CPFACCCY',	'CPFABJLT',	'CPFAASNM',	'CPFZHIG',	'CPFAAPTS',	'CUSNAH',	'CPFAONG',	'CPFAATKW',	'CPFAADCY',	'CPFAAMZN']

# COMMAND ----------

# MAGIC %md
# MAGIC #### read data

# COMMAND ----------

# MAGIC %md
# MAGIC ##### read in data from BIX
# MAGIC

# COMMAND ----------

sc_df = spark.read.table("brz_sen_norm.bix_ml_servicecase_replyhistory_cpfb_cug_nice_work_genericcase") \
    .select("frontliner_workbasket", "team_type", "department", "assigned_oic", "case_id", "created_date", "receipt_date","closed_date")

# COMMAND ----------

sc_df = sc_df.filter(sc_df.frontliner_workbasket.contains("CCU"))

# COMMAND ----------

from pyspark.sql.functions import to_date, col

sc_df = sc_df.withColumn("closed_date2", to_date("closed_date")) \
             .withColumn("receipt_date2", to_date("receipt_date"))

sc_df = sc_df.where(col("receipt_date2") >= "2022-01-01") \
    .drop("receipt_date", "closed_date") \
    .withColumnRenamed("closed_date2", "closed_date") \
    .withColumnRenamed("receipt_date2", "receipt_date")

# COMMAND ----------

from pyspark.sql.functions import max as spark_max, min as spark_min

max_date = sc_df.select(spark_max("receipt_date")).collect()[0][0]
min_date = sc_df.select(spark_min("receipt_date")).collect()[0][0]

print(max_date)
print(min_date)

# COMMAND ----------

sc_df.display()

# COMMAND ----------

print(sc_df.count())

# logic: filter rows where assigned_oic is in ccu_agents OR department == "None" AND "frontliner workbasket" does not contain "Service Centre"
sc_df = sc_df[
    (sc_df['assigned_oic'].isin(ccu_agents)) | 
    ((sc_df['department'] == "None") & (~sc_df['frontliner_workbasket'].contains('Service Centre')))
]

print(sc_df.count())

# drop duplicates based on 'case_id'
sc_df = sc_df.drop_duplicates(subset=['case_id'])
print(sc_df.count())

# COMMAND ----------

open_count_daily_nice1 = sc_df.groupBy("receipt_date").count()

# COMMAND ----------

from pyspark.sql.functions import sequence, min, max, col, explode, lit

#### Fill in missing dates ####

# Generate all dates in the range
all_dates_df = spark.createDataFrame([(min_date, max_date)], ["start", "end"]) \
    .select(explode(sequence(col("start"), col("end"))).alias("receipt_date"))

# Left join to find missing dates
missing_dates_df = all_dates_df.join(open_count_daily_nice1, on="receipt_date", how="left_anti") \
    .withColumn("count", lit(0))

# Union missing dates with original grouped DataFrame
open_count_daily_nice1 = open_count_daily_nice1.unionByName(missing_dates_df)

display(open_count_daily_nice1)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### read data from NICE 2.0

# COMMAND ----------

svr_case_2 = spark.read.table("brz_sen_norm.service_case")
nice2_data = svr_case_2.select("casenumber", "crm_receipt_date__c")

# COMMAND ----------

nice2_data = nice2_data.filter((col("crm_department__c") == "CCC") & (col("origin") != "Phone"))

# COMMAND ----------

# child case
# from pyspark.sql.functions import when, col,count
# svr_case_filtered = svr_case_filtered.withColumn("is_child", when(col("parent_casenumber").isNotNull(), 1).otherwise(0))
# display(svr_case_filtered)
# is_child_counts = svr_case_filtered.groupBy("is_child").agg(count("*").alias("row_count"))
# display(is_child_counts)

# COMMAND ----------

from pyspark.sql.functions import count, to_date, substring

nice2_data = nice2_data.withColumn("crm_receipt_date__c", to_date(substring("crm_receipt_date__c", 1, 10), "yyyy-MM-dd"))
open_count_daily_nice2 = nice2_data.groupBy("crm_receipt_date__c").agg(count("*").alias("count"))
#display(closed_count_daily_nice12)

# COMMAND ----------

open_count_daily_nice2 = open_count_daily_nice2.withColumnRenamed("crm_receipt_date__c", "receipt_date")
open_count_daily_nice2 = open_count_daily_nice2.filter((col("receipt_date") <= "2025-06-30") & (col("receipt_date") > "2024-06-21"))

# COMMAND ----------

from pyspark.sql.functions import max as spark_max, min as spark_min

max_date = open_count_daily_nice2.select(spark_max("receipt_date")).collect()[0][0]
min_date = open_count_daily_nice2.select(spark_min("receipt_date")).collect()[0][0]

print(max_date)
print(min_date)

# COMMAND ----------

from pyspark.sql.functions import sequence, min, max, col, explode, lit

#### Fill in missing dates ####

# Generate all dates in the range
all_dates_df = spark.createDataFrame([(min_date, max_date)], ["start", "end"]) \
    .select(explode(sequence(col("start"), col("end"))).alias("receipt_date"))

# Left join to find missing dates
missing_dates_df = all_dates_df.join(open_count_daily_nice2, on="receipt_date", how="left_anti") \
    .withColumn("count", lit(0))

# Union missing dates with original grouped DataFrame
open_count_daily_nice2 = open_count_daily_nice2.unionByName(missing_dates_df).orderBy("receipt_date")
display(open_count_daily_nice2)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### combine NICE 1 and 2

# COMMAND ----------

combined_open = open_count_daily_nice1.unionByName(open_count_daily_nice2)
display(combined_open)

# COMMAND ----------

combined_open_pd = combined_open.toPandas() # convert to pandas
combined_open_pd.size

# COMMAND ----------

# MAGIC %md
# MAGIC ##### patch historical missing data

# COMMAND ----------

combined_open_pd["receipt_date"] = pd.to_datetime(combined_open_pd['receipt_date'])

# taken from another csv to patch missing values 
actual_case_count_1 = [702, 605, 585, 507, 486, 283, 234, 573, 569, 679, 620, 552, 294, 299, 636,
                       569, 591, 550, 529, 272, 253, 551, 609, 565, 588, 463, 221, 238, 581, 514,
                       510, 494, 474, 255, 231, 540, 527, 545, 469]
date_1 = pd.date_range(start="2022-02-21", end="2022-03-31", freq='d').tolist()
patch_df_1a = pd.DataFrame({"receipt_date": date_1, "count": actual_case_count_1})

actual_case_count_2 = [453, 241, 197, 549, 535]
date_2 = pd.date_range(start="2022-06-24", end="2022-06-28", freq='d').tolist()
patch_df_1b = pd.DataFrame({"receipt_date": date_2, "count": actual_case_count_2})

actual_case_count_3 = [166, 251, 587, 541, 489, 511, 245, 248, 572, 523, 509, 609, 508, 234, 194, 676]
date_3 = pd.date_range(start="2022-07-10", end="2022-07-25", freq='d').tolist()
patch_df_1c = pd.DataFrame({"receipt_date": date_3, "count": actual_case_count_3})

all_patches = pd.concat([patch_df_1a, patch_df_1b, patch_df_1c])

combined_open_pd_patched = combined_open_pd.merge(all_patches, on='receipt_date', how='left', suffixes=('', '_patched'))
combined_open_pd_patched['count'] = combined_open_pd_patched['count_patched'].combine_first(combined_open_pd_patched['count'])
combined_open_pd_patched = combined_open_pd_patched.drop(columns=['count_patched'])

print(combined_open_pd_patched.size)


# COMMAND ----------

combined_open_pd_patched.display()

# COMMAND ----------

def patch_data(df, patch_rules):
    """
    df: Pandas DataFrame with 'receipt_date' (datetime64[ns]) and 'count' (int)
    patch_rules: List of tuples (target_start, target_end, source_start, source_end) target --> where u want to patch, source --> which patched dates to take from
                 All dates are strings in 'yyyy-MM-dd' format

    patch logic: taking data from the week before 
    """
    df = df.copy()

    for i, (target_start, target_end, source_start, source_end) in enumerate(patch_rules, 1):
        target_start_date = datetime.strptime(target_start, '%Y-%m-%d')
        target_end_date = datetime.strptime(target_end, '%Y-%m-%d')
        source_start_date = datetime.strptime(source_start, '%Y-%m-%d')
        source_end_date = datetime.strptime(source_end, '%Y-%m-%d')

        # Filter and sort
        target_df = df[(df["receipt_date"] >= target_start_date) & (df["receipt_date"] <= target_end_date)].copy()
        source_df = df[(df["receipt_date"] >= source_start_date) & (df["receipt_date"] <= source_end_date)].copy()

        target_df = target_df.sort_values("receipt_date").reset_index(drop=True)
        source_df = source_df.sort_values("receipt_date").reset_index(drop=True)

        len_target = len(target_df)
        len_source = len(source_df)

        # # diagnostic if mismatch
        # if len_target != len_source:
        #     print(f"  Mismatch in rule {i}:")
        #     print(f"    Target range: {target_start} to {target_end} — {len_target} rows")
        #     print(f"    Source range: {source_start} to {source_end} — {len_source} rows")
        #     print("    Missing target dates:" if len_target < len_source else "    Missing source dates:")
        #     print("    ", set(pd.date_range(target_start_date, target_end_date)) - set(target_df["receipt_date"])
        #            if len_target < len_source else
        #            set(pd.date_range(source_start_date, source_end_date)) - set(source_df["receipt_date"]))
        #     print()
        #     continue  # skip this rule

        # Patch
        mask = (df["receipt_date"] >= target_start_date) & (df["receipt_date"] <= target_end_date)
        df.loc[mask, "count"] = source_df["count"].values

    return df


# COMMAND ----------

# patch rules refer to timeseries graph
patch_rules = [
    ('2024-06-21', '2024-06-29', '2024-06-07', '2024-06-14'),
    ('2023-12-01', '2023-12-10', '2023-12-17', '2023-12-26'),
    ('2024-05-11', '2024-05-12', '2024-05-04', '2024-05-05')
]

# apply patch
combined_open_pd_patched_2 = patch_data(combined_open_pd_patched, patch_rules)
print(f"Number of days: {len(combined_open_pd_patched_2)}")

# COMMAND ----------

combined_open_pd_patched_2.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### params

# COMMAND ----------

# MAGIC %md
# MAGIC ##### CV period
# MAGIC

# COMMAND ----------

# define cross validation period
cv_start  = pd.to_datetime("2022-01-1")
cv_end    = pd.to_datetime("2025-04-30")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Test period
# MAGIC

# COMMAND ----------

# define cross validation period
test_start  = pd.to_datetime("2025-05-1")
test_end    = pd.to_datetime("2025-06-30")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### get holidays

# COMMAND ----------

sg_holidays = holidays.country_holidays('Singapore')
list_of_dates = pd.date_range(start="2020-11-01",end="2026-12-31").tolist()

holidays.country_holidays('Singapore').get('2014-01-01')
#holidays.country_holidays('Singapore').get_list([str(PH) for PH in list_of_dates if PH in sg_holidays ] )
sg_ph = pd.DataFrame( data = { "date": [PH for PH in list_of_dates if PH in sg_holidays ] ,     "holiday" : [sg_holidays.get(PH) for PH in list_of_dates if PH in sg_holidays ] })
sg_ph["is_ph"] = [1] * len(sg_ph)

# remove the string from the column using the regular expression pattern
sg_ph["holiday"]= sg_ph["holiday"].str.replace(re.compile(r'\(Observed\)'), '')
sg_ph["holiday"]= sg_ph["holiday"].str.replace(re.compile(r'\* \(\*estimated\)'), '')
sg_ph["holiday"]= sg_ph["holiday"].str.rstrip().str.lstrip()
display(sg_ph)


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Define error metric
# MAGIC

# COMMAND ----------

#weighted MAPE 
def calculate_kpi(predicted, actual, use_weight):
  error = actual - predicted
  
  if use_weight:
      weights_ =  actual/ sum(actual) 
      return 100 * sum( abs(error / (actual+0.1)) * weights_ ) 
  else:
      return 100 * np.mean( abs(error / (actual+0.1))) 
    
calculate_kpi( pd.Series([100,100]), pd.Series([110,120]),use_weight  = False)
#calculate_kpi( c(100,100), c(110,120),use_weight  = F)

# COMMAND ----------

all_fcst = pd.DataFrame()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Models
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Arimax
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###### set up the exogenous (external) regressors 
# MAGIC

# COMMAND ----------

# create exo regressors 
combined_regressors = combined_open_pd_patched_2.copy()
combined_regressors["receipt_date"] = pd.to_datetime(combined_regressors["receipt_date"])

# appead 60 day horizon future date to df for prediction 
last_date = combined_regressors["receipt_date"].max()
future = pd.DataFrame({
    "receipt_date": pd.date_range(start=last_date, end=last_date + timedelta(100), freq='D'),
    "count": [0] * 101
})
# combined_regressors.display()
# future.display()
combined_regressors = pd.concat([combined_regressors, future], ignore_index=True)

# merge in public holiday information 
combined_regressors = combined_regressors.merge(sg_ph, how="left", left_on="receipt_date", right_on="date")
combined_regressors["is_ph"] = combined_regressors["is_ph"].fillna(0)
combined_regressors.drop(["date"], axis=1, inplace=True)

# other calender based features 
combined_regressors['is_ph_lag1'] = combined_regressors['is_ph'].shift(1, fill_value=0)
combined_regressors['wday'] = combined_regressors['receipt_date'].dt.dayofweek

# for Mondays that are not holidays, but follow a holiday on Friday or Saturday, we set a special flag (helper = 1).
combined_regressors['helper'] = np.where(
    (combined_regressors["wday"] == 0) &
    ((combined_regressors["is_ph"].shift(3, fill_value=0) == 1) | (combined_regressors["is_ph"].shift(2, fill_value=0) == 1)) &
    (combined_regressors["is_ph"] != 1),
    1, 0)

# readjust is_ph_lag
combined_regressors['is_ph_lag1'] = np.where(
    combined_regressors["helper"] == 1,
    1,
    combined_regressors["is_ph_lag1"]
)

# If today is a holiday, don’t need lag info, If it's a weekend,  suppress is_ph_lag1 = 1, because that lag should affect Monday instead.
combined_regressors['is_ph_lag1'] = np.where(
    ((combined_regressors["wday"] == 5) | (combined_regressors["wday"] == 6)) &
    (combined_regressors["is_ph_lag1"] == 1),
    0,
    combined_regressors["is_ph_lag1"])

combined_regressors['is_ph_lag1'] = np.where(
    combined_regressors["is_ph"] == 1,
    0,
    combined_regressors["is_ph_lag1"])

# add yearday for seasonality
combined_regressors["yearday"] = combined_regressors['receipt_date'].dt.dayofyear

combined_regressors.display()

# COMMAND ----------

# Parameters for rolling window CV
initial_train_period = 30  # days for training window length
horizon = 30                # days to forecast/test
period = 30                 # rolling window step size in days

# Generate rolling window fold dates
rolling_starts = []
rolling_current_train_start = cv_start

while True:
    rolling_train_end = rolling_current_train_start + pd.Timedelta(days=initial_train_period)
    rolling_test_start = rolling_train_end
    rolling_test_end = rolling_test_start + pd.Timedelta(days=horizon)

    if rolling_test_end > cv_end:
        break

    rolling_starts.append((
        rolling_current_train_start,
        rolling_train_end,
        rolling_test_start,
        rolling_test_end
    ))

    rolling_current_train_start += pd.Timedelta(days=period)

print(f"Number of CV folds: {len(rolling_starts)}")
rolling_starts

# COMMAND ----------

# MAGIC %md
# MAGIC ###### cv 

# COMMAND ----------

# import pandas as pd
# import numpy as np
# import mlflow
# import mlflow.sklearn
# from pmdarima import auto_arima
# from sklearn.metrics import (
#     mean_absolute_error,
#     mean_squared_error,
#     mean_absolute_percentage_error
# )
# import time as t

# fold = 0
# arima_fcst_cv = pd.DataFrame()

# # Time Series CV with ARIMA
# for rolling_train_start, rolling_train_end, rolling_test_start, rolling_test_end in rolling_starts:
#     fold += 1

#     # Get training and testing sets based on the rolling window (IMPT: sort and set index)
#     rolling_train_df = combined_regressors[
#         (combined_regressors['receipt_date'] >= rolling_train_start) &
#         (combined_regressors['receipt_date'] < rolling_train_end)
#     ].sort_values('receipt_date').reset_index(drop=True)
#     rolling_train_df.set_index('receipt_date', inplace=True)

#     rolling_test_df = combined_regressors[
#         (combined_regressors['receipt_date'] >= rolling_test_start) &
#         (combined_regressors['receipt_date'] < rolling_test_end)
#     ].sort_values('receipt_date').reset_index(drop=True)
#     rolling_test_df.set_index('receipt_date', inplace=True)

#     # Split into features and target
#     X_train = rolling_train_df[['is_ph', 'is_ph_lag1', 'wday', 'yearday']]
#     y_train = rolling_train_df['count']

#     X_test = rolling_test_df[['is_ph', 'is_ph_lag1', 'wday', 'yearday']]
#     y_test = rolling_test_df['count']

#     time_start = t.time()

#     # Train ARIMA model
#     arima_model = auto_arima(
#         y_train,
#         exogenous=X_train,
#         seasonal=True,
#         m=7,
#         stepwise=True,
#         suppress_warnings=True,
#         error_action='ignore',
#         trace=False
#     )

#     # Predict
#     forecast = arima_model.predict(n_periods=len(y_test), exogenous=X_test)
#     arima_fcst_cv = pd.concat([
#         arima_fcst_cv,
#         pd.DataFrame({
#             'date': y_test.index,
#             'model': "arima",
#             'actual': y_test.values,
#             'forecast': forecast
#         })
#     ], axis=0)

#     # Evaluation
#     mae = mean_absolute_error(y_test, forecast)
#     rmse = np.sqrt(mean_squared_error(y_test, forecast))
#     mape = mean_absolute_percentage_error(y_test, forecast)

#     # Custom metric
#     kpi = calculate_kpi(forecast, y_test, use_weight=True)

#     elapsed = t.time() - time_start

#     # MLflow Logging
#     with mlflow.start_run(run_name=f"ARIMA_Fold_{fold}"):
#         mlflow.log_metric("MAE", mae)
#         mlflow.log_metric("RMSE", rmse)
#         mlflow.log_metric("MAPE", mape)
#         mlflow.log_metric("KPI", kpi)
#         mlflow.log_metric("elapsed_time_sec", elapsed)


# COMMAND ----------

# df_actual_pred.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### get prediction (after cv)

# COMMAND ----------

# use the final run for prediction to get the 60 days forcast

# get the test set (not part of the CV)
test_df = combined_regressors[(combined_regressors['receipt_date'] >= test_start) & (combined_regressors['receipt_date'] <= test_end)]
test_df = test_df.sort_values('receipt_date').reset_index(drop=True)
test_df.set_index('receipt_date', inplace=True)

# cross validation set
train_df = combined_regressors[(combined_regressors['receipt_date'] >= cv_start) & (combined_regressors['receipt_date'] <= cv_end)]
train_df = train_df.sort_values('receipt_date').reset_index(drop=True)
train_df.set_index('receipt_date', inplace=True)

# train and test set
X_train, y_train = train_df[['is_ph', 'is_ph_lag1', 'wday', 'yearday']], train_df['count']
X_test, y_test = test_df[['is_ph', 'is_ph_lag1', 'wday', 'yearday']], test_df['count']

# Train model
model = auto_arima(
        y_train,
        exogenous=X_train,
        seasonal=True,
        m=7,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=False)
    
# get pred 
forecast = model.predict(n_periods = len(y_test), exogenous = X_test) # n_periods == horizon

arima_fcst = pd.DataFrame({
    'date':  X_test.index,
    'model': "arima",
    'actual': y_test.values,
    'fcst': forecast
})

arima_fcst["fcst"] = round(arima_fcst["fcst"],0)
arima_fcst.display()

# COMMAND ----------

arima_kpi = calculate_kpi(arima_fcst['fcst'], arima_fcst['actual'], use_weight=True)
print(f"KPI for arima: {calculate_kpi(arima_fcst['fcst'], arima_fcst['actual'], use_weight=True)}")

# COMMAND ----------

all_fcst = pd.concat([all_fcst, arima_fcst], axis=0)

# COMMAND ----------

display(all_fcst)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Prophet
# MAGIC
# MAGIC Prophet expects the DataFrame to have:
# MAGIC - A datetime column named ds
# MAGIC - A target column named y

# COMMAND ----------

sg_ph_propht = sg_ph.copy()

sg_ph_propht["ds"] = pd.to_datetime(sg_ph["date"])
sg_ph_propht["lower_window"] = 0 
sg_ph_propht["upper_window"] = 1
sg_ph_propht = sg_ph_propht.drop(["date","is_ph"], axis = 1)
display(sg_ph_propht)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### cv (no hyperparam tunning)
# MAGIC

# COMMAND ----------

combined_regressors = combined_open_pd_patched_2.copy()
combined_regressors = combined_regressors.rename(columns={"receipt_date":"ds", "count":"y" } ) # end date 06-30-2025

# COMMAND ----------

# from prophet.diagnostics import cross_validation, performance_metrics
# import matplotlib.pyplot as plt

# prophet_model = Prophet(holidays= sg_ph_propht).fit(combined_regressors)

# # generate monthly start dates (e.g., 1st of each month)
# monthly_dates = pd.date_range(start=cv_start, end=cv_end, freq='MS')
# cutoff_dates = list(monthly_dates)

# # cv
# df_cv = cross_validation(
#     prophet_model,
#     initial='730 days',   # First training window (e.g. 2 years)
#     # period='180 days',    # Step between each CV fold
#     cutoffs=cutoff_dates,
#     horizon='60 days',   # Forecast horizon for each fold
# )

# df_metrics = performance_metrics(df_cv)
# df_metrics.display()

# #  Plot CV forecast vs actual
# plt.figure(figsize=(12, 6))
# plt.plot(df_cv['ds'], df_cv['y'], 'k.', markersize=2, label='Actual (CV)')
# plt.plot(df_cv['ds'], df_cv['yhat'], 'b-', label='Forecast (CV)')
# plt.fill_between(df_cv['ds'], df_cv['yhat_lower'], df_cv['yhat_upper'],
#                  color='skyblue', alpha=0.3, label='Uncertainty Interval (CV)')

# plt.xlabel("Date")
# plt.ylabel("Workload")
# plt.title("Prophet Cross-Validation: Forecast vs Actual")
# plt.legend()
# plt.tight_layout()
# plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ###### cv (with hyperparam tunning)

# COMMAND ----------

test_df = combined_regressors[(combined_regressors['ds'] >= test_start) & (combined_regressors['ds'] <= test_end)]
train_df = combined_regressors[(combined_regressors['ds'] >= cv_start) & (combined_regressors['ds'] <= cv_end)]
train_df.display()
test_df.display()

# COMMAND ----------

from prophet.diagnostics import performance_metrics
import time as time

# param_grid = {  
#     'changepoint_prior_scale': [0.001,0.0025,0.005, 0.01,0.025, 0.05, 0.1, 0.25, 0.5],
#     'seasonality_prior_scale': [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5, 7.5, 10.0],
#     'holidays_prior_scale'   : [0.01, 0.05,  0.1 , 0.5, 1 ,5, 10]
# }

start_time = time.time()

param_grid = {  
    'changepoint_prior_scale': [0.001],
    'seasonality_prior_scale': [0.01, 0.05],
    'holidays_prior_scale'   : [0.01]
}

# generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
latest_allowed_cutoff = cv_end - pd.Timedelta(days=60) # need to change the horizon
monthly_dates = pd.date_range(start=cv_start, end=latest_allowed_cutoff, freq='MS')  # date range of the cv_train set
cutoff_dates = list(monthly_dates)[1:]
kpis = []

for params in all_params:
    prophet_model = Prophet(**params).fit(train_df)  # df should have columns ds and y
    
    df_cv = cross_validation(
    prophet_model,
    initial='730 days',   # first training window (min 2 years)
    # period='180 days',    # step between each CV fold
    cutoffs=cutoff_dates,   # use predefined dates to be the same across all models
    horizon='60 days',   # forecast horizon for each fold
    )
    kpi = calculate_kpi(df_cv['yhat'], df_cv['y'], use_weight=True)
    kpis.append(kpi)

end_time = time.time()
print(f"Total time taken: {end_time - start_time} seconds")
prophet_total_time = end_time - start_time
results = pd.DataFrame(all_params)
results['kpi'] = kpis
prophet_best_params = all_params[np.argmin(kpis)]
prophet_kpi_cv = np.mean(kpi)

# COMMAND ----------

prophet_kpi_cv = np.mean(kpi)

# COMMAND ----------

kpis

# COMMAND ----------

# MAGIC %md
# MAGIC ###### get prediction (after cv)

# COMMAND ----------

# after cv with best hyperparams test on the test set 

print(f"Training from {cv_start} to {cv_end}, Testing on {test_start} to {test_end}")

prophet_model = Prophet(**prophet_best_params, holidays= sg_ph_propht).fit(train_df)
prophet_fcst_prediction = prophet_model.predict(test_df)
prophet_fcst = prophet_fcst_prediction[["ds", "yhat", "yhat_upper", "yhat_lower"]].rename(columns = {"ds":"fcst_date"})
prophet_fcst["model"] = "prophet"
prophet_fcst["actual"] = test_df.sort_values(by = ["ds"]).reset_index(drop = True)["y"]
prophet_fcst["fcst"] = round(prophet_fcst["yhat"], 0)
prophet_fcst.display()

# COMMAND ----------

prophet_fcst = prophet_fcst.loc[:,["fcst_date", "model", "actual", "fcst"]].rename(columns = {"fcst_date":"date"})
prophet_kpi = calculate_kpi(prophet_fcst["fcst"], prophet_fcst["actual"], use_weight=True)
prophet_fcst.display()

# COMMAND ----------

all_fcst = pd.concat([all_fcst, prophet_fcst[["date", "model", "actual", "fcst"]]], axis = 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### lightgbm
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###### feature engineering
# MAGIC

# COMMAND ----------

def create_features(df, sg_ph = sg_ph):
    df2 = df[["receipt_date", "count"]]
    
    df2 = df2.merge(sg_ph, left_on = "receipt_date" , right_on  ="date" , how = "left")

    df2["is_ph"] = df2["is_ph"].fillna(0)
    df2['is_ph_lag1'] = df2['is_ph'].shift(1,fill_value = 0)    # one day after public holiday
    df2['wday'] = df2['receipt_date'].dt.dayofweek   # Sat = 5, sunday =6
    df2['helper'] = np.where((df2["wday"] == 0 ) & (  (df2["is_ph"].shift(3,fill_value = 0) == 1 ) | (df2["is_ph"].shift(2,fill_value = 0) ==1)) & (   df2["is_ph"] != 1),1,0)#for all Mon that is not a PH, set helper to 1 if the previous Fri/Sat was a PH
    df2['is_ph_lag1'] = np.where(df2["helper"] == 1 , 1, df2["is_ph_lag1"] )  # push the helper column to is_ph_lag1 
    df2['is_ph_lag1'] = np.where( ((df2["wday"] == 5) | (df2["wday"] == 6) ) & (df2["is_ph_lag1"] ==1), 0 , df2["is_ph_lag1"] )  # if Friday is a PH, then corresponding is_ph_lag1 will be on Mon
    df2['is_ph_lag1'] = np.where( df2["is_ph"] ==1 , 0 , df2["is_ph_lag1"])   # for consecutive PHs, the second day's is_ph_lag1 should be 0 
    df2['is_weekend'] =  np.where(((df2["wday"] == 5) | (df2["wday"] == 6) ) ,1,0 ) 
    df2["count_lag1"] = df2['count'].shift(1,fill_value = 0)
    df2["count_lag2"] = df2['count'].shift(2,fill_value = 0)
    df2["count_lag3"] = df2['count'].shift(3,fill_value = 0)
    df2["count_lag4"] = df2['count'].shift(4,fill_value = 0)
    df2["count_lag5"] = df2['count'].shift(5,fill_value = 0)
    df2["count_lag6"] = df2['count'].shift(6,fill_value = 0)
    df2["count_lag7"] = df2['count'].shift(7,fill_value = 0)
    df2["count_lag8"] = df2['count'].shift(8,fill_value = 0)
    df2["count_lag9"] = df2['count'].shift(9,fill_value = 0)
    df2["count_lag10"] = df2['count'].shift(10,fill_value = 0)
    df2["count_lag11"] = df2['count'].shift(11,fill_value = 0)
    df2["count_lag12"] = df2['count'].shift(12,fill_value = 0)
    df2["count_lag13"] = df2['count'].shift(13,fill_value = 0)
    df2["count_lag14"] = df2['count'].shift(14,fill_value = 0)

    return(df2)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### cv (with hyperparam tunning)
# MAGIC

# COMMAND ----------

# define the input iterables
max_depth = range(2, 7)
min_data_in_leaf = range(10, 51, 5)
learning_rate = [0.25, 0.5, 1.0, 1.5]
objective = ["regression"]

# create a list of tuples representing all combinations of the input iterables
param_tuples = list(itertools.product(max_depth, min_data_in_leaf, learning_rate, objective))
param_df = pd.DataFrame(param_tuples, columns=["max_depth", "min_data_in_leaf", "learning_rate", "objective"])

param_df["num_leaves"] = round((2**param_df["max_depth"]) * 0.8).astype(int)
# set number of iterations to be larger for smaller values of learning_rate

param_df["num_iterations"] = np.where( param_df["learning_rate"] == 0.25 , 300 , 
                                      np.where(param_df["learning_rate"] == 0.5 , 200, 100 ))

#param_df.display()

#### adhoc, use predefined parameters, to remove this when doing actual tuning!!!! #####

param_df = pd.DataFrame([{'max_depth': 2, 'min_data_in_leaf': 10, 'learning_rate': 0.25, 'objective': 'regression', 'num_leaves': 3, 'n_estimators': 300, 'random_state': 42}])

param_df["num_iterations"] = np.where( param_df["learning_rate"] == 0.25 , 300 , 
                                      np.where(param_df["learning_rate"] == 0.5 , 200, 100 ))

param_df.display()

# COMMAND ----------

combined_regressors = combined_open_pd_patched_2.copy()
combined_regressors["receipt_date"] = pd.to_datetime(combined_regressors["receipt_date"])
test_df = combined_regressors[(combined_regressors['receipt_date'] >= test_start) & (combined_regressors['receipt_date'] <= test_end)]
test_df = create_features(test_df)

# cross validation set
train_df = combined_regressors[(combined_regressors['receipt_date'] >= cv_start) & (combined_regressors['receipt_date'] <= cv_end)]
train_df = create_features(train_df)
train_df.display()

# COMMAND ----------

# Parameters for rolling window CV
initial_train_period = 365*2  # days for training window length
horizon = 60                # days to forecast/test
period = 30                 # rolling window step size in days

# Generate rolling window fold dates
rolling_starts = []
rolling_current_train_start = cv_start

while True:
    rolling_train_end = rolling_current_train_start + pd.Timedelta(days=initial_train_period)
    rolling_test_start = rolling_train_end
    rolling_test_end = rolling_test_start + pd.Timedelta(days=horizon)

    if rolling_test_end > cv_end:
        break

    rolling_starts.append((
        rolling_current_train_start,
        rolling_train_end,
        rolling_test_start,
        rolling_test_end
    ))

    rolling_current_train_start += pd.Timedelta(days=period)

print(f"Number of CV folds: {len(rolling_starts)}")
rolling_starts

# COMMAND ----------

import numpy as np

cv_results = []
lgbm_fcst_cv = pd.DataFrame()

for idx, row in param_df.iterrows():
    params = {
        'max_depth': int(row['max_depth']),
        'min_data_in_leaf': int(row['min_data_in_leaf']),
        'learning_rate': float(row['learning_rate']),
        'objective': row['objective'],
        'num_leaves': int(row['num_leaves']),
        'n_estimators': int(row['num_iterations']),
        'random_state': 42
    }

    fold_kpis = []

    for rolling_train_start, rolling_train_end, rolling_test_start, rolling_test_end in rolling_starts:
        train_mask = (train_df["receipt_date"] >= rolling_train_start) & (train_df["receipt_date"] < rolling_train_end)
        test_mask = (train_df["receipt_date"] >= rolling_test_start) & (train_df["receipt_date"] < rolling_test_end)

        train_data = train_df.loc[train_mask]
        test_data = train_df.loc[test_mask]

        holiday_mapping = {'nan' :0, 'Deepavali':1, 'Christmas Day':2, "New Year's Day":3,
       'Chinese New Year':4, 'Good Friday':5, 'Labour Day':6, 'Hari Raya Puasa':7,
       'Vesak Day':8, 'Hari Raya Haji':9, 'National Day':10}

        train_data["holiday"] = train_data["holiday"].astype(str).map(holiday_mapping)   
        test_data["holiday"] = test_data["holiday"].astype(str).map(holiday_mapping)   

        train_dataset = lgb.Dataset(train_data.drop(["date", "receipt_date", "count"], axis=1), label=train_data["count"])
        
        X_test = test_data.drop(["date", "receipt_date", "count"], axis=1)
        y_test = test_data["count"]


        if len(X_train) == 0 or len(X_test) == 0:
            continue

        # train model with params and num_boost_round
        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=1000,
        )

        # Predict on test data
        preds = model.predict(X_test)
            # Predict
    
        # lgbm_fcst_cv = pd.concat([
        #     lgbm_fcst_cv,
        #     pd.DataFrame({
        #         'date': y_test.index,
        #         'model': "lgbm",
        #         'actual': y_test.values,
        #         'forecast': forecast
        #     })
        # ], axis=0)

        kpi_score = calculate_kpi(predicted=preds, actual=y_test.values, use_weight=True)
        fold_kpis.append(kpi_score)

    avg_kpi = np.mean(fold_kpis) if fold_kpis else np.inf
    cv_results.append({'params': params, 'avg_kpi': avg_kpi})
    print(f"Params idx {idx}: {params} | Avg KPI: {avg_kpi:.4f}")

sorted_list = sorted(cv_results, key=lambda x: x['avg_kpi'])
lightgbm_best_params = sorted_list[0]['params']
lightgbm_best_kpi = sorted_list[0]['avg_kpi']

# COMMAND ----------

print("\nBest hyperparameters found:")
print(lightgbm_best_params)
print(f"Best average KPI: {lightgbm_best_kpi:.4f}")

lgbm_kpi_cv = np.mean([result['avg_kpi'] for result in cv_results])
print(f"Average KPI: {lgbm_kpi_cv:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###### get prediction (after cv)
# MAGIC

# COMMAND ----------

print(f"Training from {cv_start} to {cv_end}, Testing on {test_start} to {test_end}")

X_train, y_train = train_df.drop(["date", "receipt_date", "count"], axis=1), train_df["count"]

holiday_mapping = {'nan' :0, 'Deepavali':1, 'Christmas Day':2, "New Year's Day":3,
       'Chinese New Year':4, 'Good Friday':5, 'Labour Day':6, 'Hari Raya Puasa':7,
       'Vesak Day':8, 'Hari Raya Haji':9, 'National Day':10}

X_train["holiday"] = X_train["holiday"].astype(str).map(holiday_mapping)   

lgbm_model = lgb.LGBMRegressor(**lightgbm_best_params)
lgbm_model.fit(X_train, y_train)

X_test, y_test = test_df.drop(["date", "receipt_date", "count"], axis=1), test_df["count"]
holiday_mapping = {'nan' :0, 'Deepavali':1, 'Christmas Day':2, "New Year's Day":3,
       'Chinese New Year':4, 'Good Friday':5, 'Labour Day':6, 'Hari Raya Puasa':7,
       'Vesak Day':8, 'Hari Raya Haji':9, 'National Day':10}

X_test["holiday"] = X_test["holiday"].astype(str).map(holiday_mapping)   

preds = lgbm_model.predict(X_test)

# calculate KPI
kpi_score = calculate_kpi(predicted=preds, actual=y_test.values, use_weight=True)
print(f"KPI score: {kpi_score:.4f}")


# COMMAND ----------

lgbm_fcst = pd.DataFrame()
lgbm_fcst["date"] = test_df["receipt_date"]
lgbm_fcst["model"] = "lightgbm"
lgbm_fcst["actual"] = test_df["count"]
lgbm_fcst["fcst"] = preds
lgbm_fcst["fcst"] = round(lgbm_fcst["fcst"],0)

lgbm_fcst.display()

lgbm_kpi = calculate_kpi(predicted=preds, actual=y_test.values, use_weight=True)
print(f"KPI score: {lgbm_kpi:.4f}")

# COMMAND ----------

all_fcst = pd.concat([all_fcst, lgbm_fcst], axis=0)
all_fcst.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### end of model training
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### create hybrid model 

# COMMAND ----------

all_fcst_pivot = all_fcst.pivot_table(index='date', columns='model', values='fcst', aggfunc= "mean") 
all_fcst_pivot['hybrid'] = round(all_fcst_pivot.mean(axis=1),0)
actuals = all_fcst[['date', 'actual']].drop_duplicates(subset=["date"]).set_index('date')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Final forecast
# MAGIC

# COMMAND ----------

all_fcst_pivot = all_fcst_pivot.join(actuals)
all_fcst_pivot = all_fcst_pivot.reset_index()

# coerce negative forecasts to zero
cols_to_coerce = ['arima', 'lightgbm', 'prophet', 'hybrid']
all_fcst_pivot[cols_to_coerce] = all_fcst_pivot[cols_to_coerce].clip(lower=0)

all_fcst_pivot.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### plot the forecast
# MAGIC

# COMMAND ----------

# plt.figure(figsize=(14, 6))

# # Plot each model's forecast as a line
# model_cols = ['arima', 'lightgbm', 'prophet', 'hybrid']
# colors = ['blue', 'green', 'orange', 'red']

# for model, color in zip(model_cols, colors):
#     plt.plot(all_fcst_pivot['date'], all_fcst_pivot[model], label=model, color=color)

# # actual values are the black dots
# plt.scatter(all_fcst_pivot['date'], all_fcst_pivot['actual'], color='black', label='actual', zorder=5)

# plt.xlabel('Date')
# plt.ylabel('Count')
# plt.title('Forecast vs Actual')
# plt.legend()
# # plt.grid(True)
# plt.tight_layout()

# plt.show()

# COMMAND ----------

import plotly.graph_objs as go

model_cols = ['arima', 'lightgbm', 'prophet', 'hybrid']
colors = ['blue', 'green', 'orange', 'red']

fig = go.Figure()

for model, color in zip(model_cols, colors):
    fig.add_trace(go.Scatter(
        x=all_fcst_pivot['date'],
        y=all_fcst_pivot[model],
        mode='lines',
        name=model,
        line=dict(color=color)
    ))

fig.add_trace(go.Scatter(
    x=all_fcst_pivot['date'],
    y=all_fcst_pivot['actual'],
    mode='markers',
    name='actual',
    marker=dict(color='black')
))

fig.update_layout(
    title='Forecast vs Actual',
    xaxis_title='Date',
    yaxis_title='Count',
    legend_title='Model',
    width=1000,
    height=400
)

p = plot(fig,output_type='div')
displayHTML(p)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### evaluation metrics
# MAGIC

# COMMAND ----------

print(f"KPIS of all models for test set:")
print(f"prophet: {prophet_kpi}, arima: {arima_kpi}, lgbm_kpi: {lgbm_kpi}. hybrid_kpi: {calculate_kpi(predicted=all_fcst_pivot['hybrid'], actual=all_fcst_pivot['actual'], use_weight=True)} \n")

print(f"KPIS of models for cross validation:")
print(f"prophet: {prophet_kpi_cv},  lgbm_kpi: {lgbm_kpi_cv} \n") # arima: {arima_kpi_cv},

# COMMAND ----------

import time
#time.sleep(48 * 60 * 60)
