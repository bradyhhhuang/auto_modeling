#!/usr/bin/env python
# coding: utf-8

# # Outline
# ### Set Environment
# import sources
# 
# ### Prepare Data
# **Goal: get the dataset and directories ready**
# 
# - load data
# - split data into scorecards (DataSplitter)
# - create directory (DirectoryManager)
# - create Datasets (Dataset)
# 
# ### Select Features
# **Goal: get a short list of features**
# 
# - limit range (Rangelimiter)
# - analyze features 
#     - bin data (Binner)
#     - feature profiling (XProfiler)
# - reload data with selected features
#     - create customized schema in excel csv
# 
# ### Transform Data
# **Goal: feature engineering**
# 
# - limit range (Rangelimiter)
# - bin data (Binner)
# - impute features (Imputer)
# - label WOE value (LabelTransformer)
# - save proprocessors (Pipeline)
# - *Load & Transform Additional Dataset (if available)*
# 
# ### Build Models with hyperparameters
# - set parameters
# - fit a model (Modeler)
# - compare model performance (ModelerRanker)
# 
# ### Doucument Model & Create Scoring Function
# - document data preprocessing steps
# - create scoring functions

# # Set Environment

# **RUN:** import required modules / classes and initialize jupyter notebook environment

# In[ ]:


import os 
#os.chdir('../..')

import copy
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection._search import ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sources.data_set import DataSet 
from sources.data_splitter import DataSplitter    
from sources.data_transformation import RangeLimiter
from sources.imputer import Imputer
from sources.feature_profiling import XProfiler
from sources.modeler import Modeler, ModelerRanker
from sources.label_transformer import LabelTransformer 
from sources.file_controller import FileController
from sources.console_settings import ConsoleSettings
from sources.binner import Binner 
from sources.feature_profiling import LiftPlotter  
from sources.directory_manager import DirectoryManager
from sources.analyzer import Analyzer

if __name__ =='__main__': 

    settings = ConsoleSettings()
    settings.set_all()
    __spec__ = None


# # Prepare Data

# ## Load Data

# **RUN:** load modeling & validation dataframes from csv files

# In[ ]:


m_filepath = 'Data\\MODEL_PERSON_T.csv'
v_filepath = 'Data\\MODEL_PERSON_V.csv'    

m_df = pd.read_csv(m_filepath, encoding='big5')
v_df = pd.read_csv(v_filepath, encoding='big5')


# **OUTPUT:** modeling_dataframe, validation_dataframe
# 
# **CHECK POINT:** `number_of_samples` should align with row counts, whereas `number_of_features` aligns with feature counts in the original data. 
# - ``modeling_dataframe.shape``: (number_of_samples, number_of_features)

# In[ ]:


print(m_df.shape)
print(v_df.shape)


# ## Split Dataset into Scorecards

# `DataSplitter` can be used to split dataframe objects into different dataframes and store in a dictionary with the according name of the scorecard. 
# 
# **FIT**
# 
# There are two fitting methods to choose from: `fit_numeric_rules` or `fit_nominal_rules`, with similar structure:
# - `feature`: the feature used to segment the data.
# - `criteria`: the boundry, or the assigned value of `feature` in each scorecard. 

# **FIT-numeric:**
# 
# `fit_numeric_rules(feature, criteria)`
#     
# - `criteria`
#     - Type: `dict` 
#     - Format: {`str`(card_name): `int`(maximum feature value)}

# In[ ]:


data_splitter = DataSplitter()
data_splitter.fit_numeric_rules('HOL_P09_0240', {'low': 0, 
                                                 'medium': 500000,
                                                 'high':'else'
                                                })


# **FIT-nominal:**
# 
# `fit_nominal_rules(feature, criteria)`
#     
# - `criteria`
#     - Type: `dict` 
#     - Format: {`str`(card_name): `str` or `list`(assigned feature value)}

# In[ ]:


data_splitter = DataSplitter()
data_splitter.fit_nominal_rules('L1_PATH_NEW', {'unsec_both':['UNSEC-BOTH','UNSEC-BOTH-SEC'],
                                            'cc':['CC_TXN','UNSEC-CC'],
                                            'no_cc_ln':'NO_CC_LN',
                                            'unsec':'else'
                                            })        


# **TRANSFORM:** apply the rules to split data.
# 
# `transform(dataframe)`

# In[ ]:


m_df_dict = data_splitter.transform(m_df)
v_df_dict = data_splitter.transform(v_df)


# **OUTPUT:** modeling_dataframe_dictionary, validation_dataframe_dictionary
# 
# **CHECKPOINT:** 
# 
# `number_of_samples` represents the sample counts in each scorecard, where `number_of_features` should equal to (original feature counts +1)
# - card_name: (`number_of_samples in the scorecard`, `number_of_features`)

# In[ ]:


print('===== Modeling =====')
for scorecard in m_df_dict.keys():
    print(scorecard, ":", m_df_dict[scorecard].shape)
    
print('===== Validation =====')
for scorecard in v_df_dict.keys():
    print(scorecard, ":", v_df_dict[scorecard].shape)


# **SAVE:** save `DataSplitter` object for future scoring. Execute after directory was built. (See code below after "Create Directories")

# ## Create Directory
# 
# to store objects that will be generated in the following steps.
# 
# **UNDER \Models\:**
# objects for future scoring, including:
# - data_schema 
# - data_preprocessors:`DataSplitter`,`RangeLimiter`, `binner`, `NominalImputer`,`NumericImputer`, `LabelTransformer` 
# 
# **UNDER \Docs\:**
# documents for performance review and model documentation, including:
# - charts after binning (pdf)
# - documentation of binning rules (csv)
# 
# 
# 

# `DirectoryManager` has 2 functions:
# - `make_directory(path, option)`:
#     - `path`: can be absolute path or relative path. Error will be raised if using the path that does not exist. 
#     - `option`: 
#         - `replace`: If the directory already exists, clean up all the contents in the directory.
#         - `pass`: Do nothing to the existed directory.
# - `make_multiple_directories(path_list, option)`:
#     - `path_list`: `list` of paths.
#     - `option`: same as `make_directory`.

# **RUN:**

# In[ ]:


dir_manager = DirectoryManager()
dir_manager.make_directory('Models', option='pass')
dir_manager.make_directory('Docs', option='pass')


# **SAVE:** save `DataSplitter` object to \Models for future scoring.

# In[ ]:


file_controller = FileController(filepath="Models\DataSplitter.pkl")
file_controller.save(data_splitter)


# ## Create DataSets and Directories for each Scorecard

# In this step, turn dataframe objects into our self-defined object: `DataSet`. `DataSet` is the main object type we will use in feature selection process.

# In[ ]:


schema_path = 'Data\\Data_Schema.csv'
card_name = 'unsec_both'

# create scorecard directory to store scorecard-specific objects
dir_manager.make_directory('Models\\{}'.format(card_name), option='replace')
dir_manager.make_directory('Docs\\{}'.format(card_name), option='replace')

modeling_scorecard_df = m_df_dict.get(card_name)
validation_scorecard_df = v_df_dict.get(card_name)

modeling_dataset = DataSet()
modeling_dataset.load_dataset_from_df(modeling_scorecard_df, schema_filepath=schema_path)

validation_dataset = DataSet()
validation_dataset.load_dataset_from_df(validation_scorecard_df, schema_filepath=schema_path)


# In[ ]:


modeling_dataset.x.shape
modeling_dataset.y.shape


# In[ ]:


validation_dataset.x.shape
validation_dataset.y.shape


# **ANALYSIS:** check Y% for modeling & validation datasets

# In[ ]:


print(modeling_dataset.y['BAD_1Y'].sum()/modeling_dataset.y['BAD_1Y'].count())
print(validation_dataset.y['BAD_1Y'].sum()/validation_dataset.y['BAD_1Y'].count())


# # Select Features

# ## Limit Range for Each Feature
# 
# To avoid outliers influnce the results of feature analysis, limit the range of each feature first. 
# 
# `RangeLimiter` has functions: 
# - `create_range_limits`
# - `tranform`

# **FIT:** uses Winsorizing methodology to set upper / lower bounds according to feature values
# 
# `create_range_limits(dataset, method)`: `method` has 3 options 
# - `quantile_range`: Upper = q3 + 1.5*iqr , Lower = q1 - 1.5*iqr        
# - `standard_deviation`: Upper = mean + 3*standard_deviation , Lower = mean - 3*standard_deviation 
# - `auto`: Uses `quantile_range` as default, but if q3-q1=0, then use `standard_deviation`
# 
# `set_transform_features(exclude)`
# - `exclude`:the features in this list will not be transformed when applying `transform` to `DataSet`. 
# 
# Reference Site: https://heartbeat.fritz.ai/how-to-make-your-machine-learning-models-robust-to-outliers-44d404067d07

# In[ ]:


range_limiter = RangeLimiter() 
range_limiter.create_range_limits(modeling_dataset, method='auto') #options: quantile_range, standard_deviation, auto
#range_limiter.set_transform_features(exclude=['HOL_P09_10', 'HOL_P09_11'])


# **TRANSFORM:** limit ranges for each feature according to the fitted upper / lower bounds
# 
# `transform(dataset)`

# In[ ]:


modeling_dataset = range_limiter.transform(modeling_dataset)
validation_dataset = range_limiter.transform(validation_dataset)


# **OUTPUT:** `modeling_dataset`, `validation_dataset`

# **CHECKPOINT:** each feature's `min` / `max` should be within its upper / lower bounds

# In[ ]:


# fitted upper/lower bounds in the range limiter
print(range_limiter)


# In[ ]:


print(modeling_dataset.x.describe())
print(validation_dataset.x.describe())

#    print(modeling_dataset.x.max())
#    print(modeling_dataset.x.min())
#    print(validation_dataset.x.max())
#    print(validation_dataset.x.min())


# ## Analyze Features

# ### Binner:  analyze each feature's lift
# bins the original data and check lift to select features. 

# `Binner` include several functions: 
# - **FIT:** creates bins & upper_bounds for each feature according to specified methods
# - **TRANSFORM:** transforms data according to fit result
# - **PLOT:** plots lift charts for each feature

# **FIT-1:**
# 
# `fit(dataset, nominal_rules, numeric_rules)`
# 
# - `nominal_rules`: `dict` in format:`{'method':'order', 'criteria':['event_rate(%)','desc']}`
#     - `method`: how to create bins, there is 1 option
#         - `order`: order fields according to `criteria`
#         - `manual`: allows manual-defined  binning rules
#     - `criteria`: `list` in format `[criteria, ordering]` 
#         - `criteria`: which criteria follow. There are 2 options: `woe` or `event_rate(%)`
#         - `ordering`: `asc` for ascending or `desc` for descending
# 
# - `numeric_rules`: `dict` in format: `{'max_bins':10, 'method':'percentile', 'auto': 'merge', 'criteria':'woe'}`
#     - `max_bins`: the maximum bin count
#     - `method`: how to create bins, there are 3 options
#         - `percentile`: split evenly in sample percentiles. Ex: if bin_count=10, creates decile bins
#         - `range`: split evenly in feature range. Ex: if feature ranges from 1~10 and bin_count=10, bins=(-inf,1], (1,2], ...(9,10]
#         - `tree`: split with DecisionTree algorithm, where max_leaves_count=bin_count
#         - `manual`: allows manual-defined  binning rules
#     - `auto`: whether or not to auto-bin until each bins' criteria is in monotonic order
#         - `none`: don't auto-bin
#         - `merge`: start from max_bins, if criteria is not in monotonic order, then merge the bins not in order with its previous bin until criteria reaches monotonic order 
#         - `decrease`: start from max_bins, if criteria is not in monotonic order, then decrease bin_count by 1 until criteria reaches monotonic order 
#     - `criteria`: which criteria to check if ordered, there are 2 options: 
#         - `woe`
#         - `event_rate(%)`

# In[ ]:


method = 'percentile'
auto = 'merge'
criteria = 'woe'
    
binner = Binner()
binner.fit(modeling_dataset, 
               nominal_rules={'method':'order', 'criteria':['event_rate(%)','desc']}, 
               numeric_rules={'max_bins':10, 'method':method, 'auto':auto, 'criteria':criteria})


# **FIT-2:**
# 
# `fit_specific_features(dataset, **features_n_rules)`, in which `**feature_n_rules` specifies fitting methodology / manual assign bin boundaries for specific features, formatted as: 
# 
#        'HOL_P10_9': {'feature_type':'numeric',
#                        'rules':{'max_bins':10, 
#                                 'method':'range', 
#                                 'auto':'merge', 
#                                 'criteria':'woe'}}
# 
#        'HOL_P09_9': {'feature_type':'numeric',
#                       'rules':{'max_bins':None, 
#                                'method':'manual', 
#                                'auto':'none', 
#                                'criteria':{0:1245000, 1: 'else'}}}
# 
#        'CORP_TYPE_NEW': {'feature_type':'nominal',
#                           'rules':{'method':'manual', 
#                                    'criteria':{0: ['高科技電子業','穩定收入族群'], 1: 'else'}}}  

# In[ ]:


binner.fit_specific_features(modeling_dataset, 
                             **{'HOL_P10_9': {'feature_type':'numeric',
                                               'rules':{'max_bins':10, 
                                                        'method':'range', 
                                                        'auto':'merge', 
                                                        'criteria':'woe'}},
                                'HOL_P09_9': {'feature_type':'numeric',
                                                  'rules':{'max_bins':None, 
                                                           'method':'manual', 
                                                           'auto':'none', 
                                                           'criteria':{0:1245000, 1: 'else'}}},
                                'CORP_TYPE_NEW': {'feature_type':'nominal',
                                                      'rules':{'method':'manual', 
                                                               'criteria':{0: ['高科技電子業','穩定收入族群'], 1: 'else'}}} 
                                })


# **PLOT-1:**
# 
# `plot_all(dataset_dict, filepath)` analyzes a single dataset & plots lift chart to pdf
# - `dataset_dict`: `dict` object under format `{data_name: dataset}` *only allows 1 dataset
# - `filepath`: plot will be auto-waved to pdf file with this filepath 

# In[ ]:


binner.plot({'modeling':modeling_dataset}, filepath='Docs\\{0}\\Bivariable_Analysis_{1}_{2}.pdf'.format(card_name, method, auto))


# `plot_multi_datasets_all(complete_modeling_dataset, complete_validation_dataset, filepath)` analyzes multiple datasets & plots lift chart to pdf
# - `complete_modeling_dataset`: `dict` object under format `{data_name: dataset}` *only 1 modeling dataset
# - `complete_validation_dataset`: `dict` object under format `{data_name: dataset, data_name2: dataset2}`, accepts 1+ validation datasets
# - `filepath`: plot will be auto-waved to pdf file with this filepath 

# In[ ]:


binner.plot_multi_datasets(complete_modeling_dataset={'modeling':modeling_dataset},
                           complete_validation_datasets={'validation':validation_dataset},
                           filepath='Docs\\{0}\\MultiDataset_Bivariable_Analysis_{1}_{2}.pdf'.format(card_name, method, auto))


# ### XProfiler:  Correlation between X's and Y

# **ANALYZE:** check correlation heat-map to decide whether to remove highly-related X's  *may take several minutes if many features 
# - Check `Correlations`: `Pearson` and `Spearman` sections for heat map

# In[ ]:


x_profiler = XProfiler()
x_profiler.create_report(modeling_dataset)
x_profiler.to_html(filepath=r'Models\XProfileReport.html')


# # ---------------------------------------------------------------------------------------------------

# ## Reload Data with Selected Features

# **CREATE CUSTOMIZED SCHEMA:**
# - Original data_schema csv file can be found in `Data\` folder
# - Edit: for selected features, set Category='X'; for others, change Category to any other value (ex: 'X_drop')
# - Save it to `Models\card_name\` folder
# 
# **RUN:** 

# In[ ]:


card_name = 'unsec_both'
schema_path = 'Models\\{0}\\Data_Schema_{1}.csv'.format(card_name, card_name) 

modeling_scorecard_df = m_df_dict.get(card_name)
validation_scorecard_df = v_df_dict.get(card_name)

modeling_dataset = DataSet()
modeling_dataset.load_dataset_from_df(modeling_scorecard_df, schema_filepath=schema_path)

validation_dataset = DataSet()
validation_dataset.load_dataset_from_df(validation_scorecard_df, schema_filepath=schema_path)


# In[ ]:


modeling_dataset.x.shape
modeling_dataset.y.shape


# In[ ]:


validation_dataset.x.shape
validation_dataset.y.shape


# # Transform Data

# ## Limit Range for Each Feature
# `RangeLimiter`: See `Select Feature`-`Limit Range for Each Feature` description above

# **FIT:** uses Winsorizing methodology to set upper / lower bounds according to feature values
# 
# `create_range_limits(dataset, method)`

# In[ ]:


range_limiter = RangeLimiter() 
range_limiter.create_range_limits(modeling_dataset, method='auto') #options: quantile_range, standard_deviation, auto
#range_limiter.set_transform_features(exclude=['HOL_P09_10', 'HOL_P09_11'])


# **TRANSFORM:** limit ranges for each feature according to the fitted upper / lower bounds
# 
# `transform(dataset)`

# In[ ]:


modeling_dataset = range_limiter.transform(modeling_dataset)
validation_dataset = range_limiter.transform(validation_dataset)


# **OUTPUT:** `modeling_dataset`, `validation_dataset`

# In[ ]:


print(modeling_dataset.x.describe())
print(validation_dataset.x.describe())


# ## Bin Data

# ### Binner:  analyze each feature's lift
# bins the original data to make x-y relationships more linear & stable. 

# `Binner` include several functions: 
# - **FIT:** creates bins & upper_bounds for each feature according to specified methods
# - **TRANSFORM:** transforms data according to fit result
# - **PLOT:** plots lift charts for each feature

# ### Binner:  analyze each feature's lift
# bins the original data and check lift to select features. 

# `Binner` include several functions: 
# - **FIT:** creates bins & upper_bounds for each feature according to specified methods
# - **TRANSFORM:** transforms data according to fit result
# - **PLOT:** plots lift charts for each feature

# **FIT-1:**
# 
# `fit(dataset, nominal_rules, numeric_rules)`
# 
# - `nominal_rules`: `dict` in format:`{'method':'order', 'criteria':['event_rate(%)','desc']}`
#     - `method`: how to create bins, there is 1 option
#         - `order`: order fields according to `criteria`
#         - `manual`: allows manual-defined  binning rules
#     - `criteria`: `list` in format `[criteria, ordering]` 
#         - `criteria`: which criteria follow. There are 2 options: `woe` or `event_rate(%)`
#         - `ordering`: `asc` for ascending or `desc` for descending
# 
# - `numeric_rules`: `dict` in format: `{'max_bins':10, 'method':'percentile', 'auto': 'merge', 'criteria':'woe'}`
#     - `max_bins`: the maximum bin count
#     - `method`: how to create bins, there are 3 options
#         - `percentile`: split evenly in sample percentiles. Ex: if bin_count=10, creates decile bins
#         - `range`: split evenly in feature range. Ex: if feature ranges from 1~10 and bin_count=10, bins=(-inf,1], (1,2], ...(9,10]
#         - `tree`: split with DecisionTree algorithm, where max_leaves_count=bin_count
#         - `manual`: allows manual-defined  binning rules
#     - `auto`: whether or not to auto-bin until each bins' criteria is in monotonic order
#         - `none`: don't auto-bin
#         - `merge`: start from max_bins, if criteria is not in monotonic order, then merge the bins not in order with its previous bin until criteria reaches monotonic order 
#         - `decrease`: start from max_bins, if criteria is not in monotonic order, then decrease bin_count by 1 until criteria reaches monotonic order 
#     - `criteria`: which criteria to check if ordered, there are 2 options: 
#         - `woe`
#         - `event_rate(%)`

# In[ ]:


method = 'percentile'
auto = 'merge'
criteria = 'woe'
    
binner = Binner()
binner.fit(modeling_dataset, 
               nominal_rules={'method':'order', 'criteria':['event_rate(%)','desc']}, 
               numeric_rules={'max_bins':10, 'method':method, 'auto':auto, 'criteria':criteria})


# **FIT-2:**
# 
# `fit_specific_features(dataset, **features_n_rules)`, in which `**feature_n_rules` specifies fitting methodology / manual assign bin boundaries for specific features, formatted as: 
# 
#        'HOL_P10_9': {'feature_type':'numeric',
#                        'rules':{'max_bins':10, 
#                                 'method':'range', 
#                                 'auto':'merge', 
#                                 'criteria':'woe'}}
# 
#        'HOL_P09_9': {'feature_type':'numeric',
#                       'rules':{'max_bins':None, 
#                                'method':'manual', 
#                                'auto':'none', 
#                                'criteria':{0:1245000, 1: 'else'}}}
# 
#        'CORP_TYPE_NEW': {'feature_type':'nominal',
#                           'rules':{'method':'manual', 
#                                    'criteria':{0: ['高科技電子業','穩定收入族群'], 1: 'else'}}} 

# In[ ]:


binner.fit_specific_features(modeling_dataset, 
                             **{'HOL_P10_9': {'feature_type':'numeric',
                                               'rules':{'max_bins':10, 
                                                        'method':'range', 
                                                        'auto':'merge', 
                                                        'criteria':'woe'}},
                                'HOL_P10_12': {'feature_type':'numeric',
                                                  'rules':{'max_bins':None, 
                                                           'method':'manual', 
                                                           'auto':'none', 
                                                           'criteria':{0:999, 1: 'else'}}},
                                'CORP_TYPE_NEW': {'feature_type':'nominal',
                                                      'rules':{'method':'manual', 
                                                               'criteria':{0: ['高科技電子業','穩定收入族群'], 1: 'else'}}} 
                                })


# **PLOT-1:**
# 
# `plot(dataset_dict, filepath)` analyzes a single dataset & plots lift chart to pdf
# - `dataset_dict`: `dict` object under format `{data_name: dataset}` *only allows 1 dataset
# - `filepath`: plot will be auto-waved to pdf file with this filepath 

# In[ ]:


binner.plot({'modeling':modeling_dataset}, filepath='Docs\\{0}\\Bivariable_Analysis_{1}_{2}.pdf'.format(card_name, method, auto))


# `plot_multi_datasets_all(complete_modeling_dataset, complete_validation_dataset, filepath)` analyzes multiple datasets & plots lift chart to pdf
# - `complete_modeling_dataset`: `dict` object under format `{data_name: dataset}` *only 1 modeling dataset
# - `complete_validation_dataset`: `dict` object under format `{data_name: dataset, data_name2: dataset2}`, accepts 1+ validation datasets
# - `filepath`: plot will be auto-waved to pdf file with this filepath 

# In[ ]:


binner.plot_multi_datasets(complete_modeling_dataset={'modeling':modeling_dataset},
                           complete_validation_datasets={'validation':validation_dataset},
                           filepath='Docs\\{0}\\MultiDataset_Bivariable_Analysis_{1}_{2}.pdf'.format(card_name, method, auto))


# **TRANSFORM:** 
# 
# `transform(dataset)`

# In[ ]:


modeling_dataset = binner.transform(modeling_dataset)
validation_dataset = binner.transform(validation_dataset)


# **OUTPUT:** `modeling_dataset`, `validation_dataset`
# 
# **CHECKPOINT:** X values should be binned according to fitted boundaries

# In[ ]:


print(modeling_dataset.x.head())
print(validation_dataset.x.head())


# **PROGRESS:**

# In[ ]:


print(binner)


# ## Impute

# ### Impute features

# **FIT-1:** selects a value to replace `np.nan` with specified rules
# 
# `fit(dataset, numeric_rule, numeric_value, nominal_rule, nominal_value)`
# - `nominal_rule` has 2 options: 
#     - `most_frequent`: fills `np.nan` (if any) with the value with the largest samples
#     - `constant`: fills `np.nan` (if any) with string `nominal_value`
# - `nominal_value`: default `'NaN'` if not specified
# - `numeric_rule` has 2 options: 
#     - `median`: fills `np.nan` (if any) with median
#     - `mean`: fills `np.nan` (if any) with mean
#     - `constant`: fills `np.nan` (if any) with `numeric_value`
# - `numeric_value`: default `0` if not specified

# In[ ]:


imputer = Imputer()
imputer.fit(modeling_dataset, numeric_rule="median", nominal_rule="most_frequent")


# **FIT-2:** assign values to be imputed manually 
# 
# `fit_specific_features(numeric_features, nominal_features)`
# - `numeric_features`: `dict` that contains {`feature1`:`impute_value1`, `feature2`:`impute_value2`}
# - `nominal_features`: `dict` that contains {`feature1`:`impute_value1`, `feature2`:`impute_value2`}

# In[ ]:


imputer.fit_specific_features(numeric_features={'USE_P12_0254': 0, 'INQ_P01_0012': 2}, 
                              nominal_features={'CORP_TYPE_NEW': 2})


# **TRANSFORM:** replaces `np.nan` in nominal features with fitted values
# 
# `trasnform(dataset)`

# In[ ]:


modeling_dataset = imputer.transform(modeling_dataset) 
validation_dataset = imputer.transform(validation_dataset)


# **OUTPUT:** `modeling_dataset`, `validation_dataset`
# 
# **CHECKPOINT:** all nominal features should contain no `np.nan`
# - `dataset.x.isnull().sum()` should be 0 

# In[ ]:


modeling_dataset.x.isnull().sum()
validation_dataset.x.isnull().sum()


# **PROGRESS:** shows features and its respective impute values

# In[ ]:


print(imputer)


# ## Label binned data value as WOE values

# **FIT:** calculates woe value for each bin, and stores data for bin labeling 

# In[ ]:


label_transformer = LabelTransformer()
label_transformer.fit(modeling_dataset)


# **TRANSFORM:** labels each row with its bins' woe value

# In[ ]:


modeling_dataset = label_transformer.transform(modeling_dataset)
validation_dataset = label_transformer.transform(validation_dataset)


# **OUTPUT:** `modeling_dataset`, `validation_dataset`
# 
# **CHECKPOINT:** values should be assigned to woe values 

# In[ ]:


modeling_dataset.x.head()
validation_dataset.x.head()


# ## Save preprocessors to Pipeline
# collect proprocessors into a `Pipeline` object and save it for future data preperation.

# In[ ]:


pipeline = Pipeline([
        ('range_limiter',range_limiter), 
        ('binner', binner), 
        ('imputer', imputer), 
        ('label_transformer', label_transformer)
        ])


# In[ ]:


file_controller = FileController("Models\\{0}\\Pipeline.pkl".format(card_name))
file_controller.save(pipeline)


# # Load & Transform Additional Dataset

# **SAVE:** combine all preprocessors into one single `Pipeline` & save to pkl file for future scoring

# **TRANSFORM:** follow the same steps to load & transform additional datasets

# In[ ]:


v_pl_filepath = 'Data\\MODEL_PROD_V_PL.csv'

v_pl_df = pd.read_csv(v_pl_filepath, encoding='big5')
v_pl_df_dict = data_splitter.transform(v_pl_df)

validation_pl_scorecard_df = v_pl_df_dict.get(card_name)

validation_pl_dataset = DataSet()
validation_pl_dataset.load_dataset_from_df(validation_pl_scorecard_df, schema_filepath=schema_path)

validation_pl_dataset = pipeline.transform(validation_pl_dataset)

#    validation_pl_dataset = range_limiter.transform(validation_pl_dataset)
#    validation_pl_dataset = binner.transform_all(validation_pl_dataset)
#    validation_pl_dataset = imputer.transform(validation_pl_dataset)
#    validation_pl_dataset = label_transformer.transform(validation_pl_dataset)


# **OUTPUT:** `dataset`
# 
# **CHECKPOINT:** should contain no nulls and values;  should be binned & labeled as WOE

# In[ ]:


validation_pl_dataset.x.describe()
validation_pl_dataset.x.head()


# # Build Models

# ## Build a single model

# **SET PARAMETER:**
# 
# *Function is implemented with `sklearn.linear_model.LogisticRegression()`, for detailed parameter definitions see reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

# In[ ]:


model_id = 'LogisticRegression_test'
parameter = {'C':1.0, 'class_weight':'balanced', 'n_jobs':-1, 'random_state':297, 'tol':0.0001}
clf = LogisticRegression()
clf = clf.set_params(**parameter)


# ### Create a Modeler
# `Modeler` is a customized object to train and evaluate models.
# 
# There are 4 functions in `Modeler`:
# - `fit`: train the model, bin the predicted event probability and generate performance evaluation accordingly.
# - `fit_binner`: reset the binning rules for this model, and regenerate performance evaluation.
# - `plot`: plot the binning result and output to file
# 
# There are 3 properties in `Modeler`:
# - `clf`: the fitted classifier (based on sklearn), presents the selected parameter
# - `performance_df`: presents model performance indicators, including KS, IV, PSI...
# - `feature_info`(for now, only applies to linear models): presents the predictors and their coefficient, correlation, p-value and correction 
# 
# **RUN:**

# In[ ]:


modeler = Modeler(model_id)
modeler.fit(clf, 
            modeling_dataset={'modeling':modeling_dataset},
            validation_datasets={'validation_1':validation_dataset, 'validation_2':validation_dataset},
            rules={'method':'percentile','max_bins':10, 'auto':'none','criteria':'none'})

modeler.plot(modeling_dataset={'modeling':modeling_dataset},
             validation_datasets={'validation_1':validation_dataset, 'validation_2':validation_pl_dataset},
             plot_path='Docs\{}'.format(card_name)) 


# **VIEW MODEL INFO:** 

# In[ ]:


print(modeler.clf)
print(modeler.performance_df)
print(modeler.feature_info)


# **SAVE:**

# In[ ]:


file_controller = FileController(filepath="Models\\{0}\\{1}.pkl".format(card_name,model_id))
file_controller.save(modeler)


# ## Tune hyperparameters

# ### Run Models with Hyperparameters
# **SET PARAMETERS**
# - define `param_grid`
# 
# **FIT**
# - create & fit `Modeler` object for each parameters
# - collect these `Modeler` results in a `Dict` for later model selection
# 
# *Function is implemented with `sklearn.linear_model.LogisticRegression()`, for detailed parameter definitionssee reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

# In[ ]:


param_grid = {   'penalty': ['l1','l2'] #done
#                    ,'dual': [True,False]   #"dual" does not change the result, only potentially the speed of the algorithm
                ,'tol': np.logspace(-5,1,7) #done
                ,'C': np.logspace(-3,1,5) #done
#                    ,'fit_intercept': [True,False] #done
#                    ,'intercept_scaling': [1]
                ,'class_weight': ['balanced',{0:0.01,1:0.99},{0:0.005,1:0.995},{0:0.001,1:0.999}]
                ,'random_state': [None]
#                    ,'solver': ['newton-cg','lbfgs','liblinear','sag','saga'] #done
#                    ,'max_iter': [100]
#                    ,'multi_class': ['ovr','multinomial']
#                    ,'verbose': [0]
#                    ,'warm_start': [True,False] #doesn't impact much 
                ,'n_jobs': [-1]
             }    
param_combinations = list(ParameterGrid(param_grid))

n=0
modelers = {}

for parameters in product(param_combinations):
    n += 1
    model_id = 'LogistictRegression_'+str(n)
    clf = LogisticRegression()    
    clf = clf.set_params(**parameters[0])
    
    modeler = Modeler(model_id)
    modeler.fit(clf, 
            modeling_dataset={'modeling':modeling_dataset},
            validation_datasets={'validation_1':validation_dataset, 'validation_2':validation_pl_dataset},
            rules={'method':'percentile','max_bins':10, 'auto':'none','criteria':'none'})

    modelers.update({model_id:modeler})


# In[ ]:


print(modelers.keys())


# ### Compare Model Performance
# Initiate a `ModelerRanker` to find the best performance model.
# 
# **SET FILTER & ORDER CRITERION:** 
# - `set_filters`: only models that qualify all the filter criterion will be ranked 
#     - format: 
#             {dataset_1: {indicator_1 : [operator, limit_value], indicator_2:[operator, limit_value]},
#             dataset_2:{indicator_1 : [operator, limit_value], indicator_2:[operator, limit_value]}}
# - `set_orders`: rank models by indicators
#     - format:
#             {priority_1:[dataset_1, indicator, ascending_order],
#             priority_2:[dataset_2, indicator, ascending_order]}
# - `rank`: apply filter and order rules to all the `modeler`s in the dictionary and output a ranking table
# 
# 
# **AVAILABLE INDICATORS**
# - `KS`: KS value for specific dataset, the higher the better 
# - `IV`: IV value for specific dataset, the higher the better 
# - `bounce_cnt`: how many times that the ordering of lift has bounced, the less the better
# - `bounce_pct`: bounce_pct% = sum(bounce_range)/total_range, the less the better 
# - `r`: correlation coefficient between bin number and respond (Y)

# In[ ]:


filters = {'modeling':
            {'ks': ['>', 50],
             'iv':['>', 100],
             'bounce_cnt':['<=', 3],
             'bounce_pct':['<', 20],
             'r':['>', 0.1]}
            ,
           'validation_1':
            {'ks': ['>', 40],
             'iv':['>', 90],
             'bounce_cnt':['<=', 5],
             'bounce_pct':['<=', 30],
             'r':['>=', 0.1]}                 
           }
                
orders = { '1':['validation_1','ks','desc']
          ,'2':['modeling','ks','desc']
          ,'3':['validation_2','ks', 'desc']
        }

model_ranker = ModelerRanker()        
model_ranker.set_filters(**filters)
model_ranker.set_orders(**orders)  
ranking_table = model_ranker.rank(modelers)      
print(ranking_table)


# **ANALYZE:** 
# 
# `top_n_models` can be used to get a list of model_ids within the assigned rank. Then utilize a loop to print out the plots and compare these models

# In[ ]:


top_n_list = model_ranker.top_n_models(n=5)

for model in top_n_list:
    modeler = modelers.get(model)
    print(modeler.clf)
    print(modeler.feature_info)
    modeler.plot(modeling_dataset={'modeling':modeling_dataset},
                 validation_datasets={'validation_1':validation_dataset, 'validation_2':validation_pl_dataset},
                 plot_path = 'Docs\{}'.format(card_name))


# # Document Model & Create Scoring Function

# ## Document Data Preprocessing Steps 

# **Reload Datasets**

# In[ ]:


m_df = pd.read_csv(m_filepath, encoding='big5')
v_df = pd.read_csv(v_filepath, encoding='big5')

m_df_dict = data_splitter.transform(m_df)
v_df_dict = data_splitter.transform(v_df)

modeling_scorecard_df = m_df_dict.get(card_name)
validation_scorecard_df = v_df_dict.get(card_name)

modeling_dataset_for_doc = DataSet()
modeling_dataset_for_doc.load_dataset_from_df(modeling_scorecard_df, schema_filepath=schema_path)

validation_dataset_for_doc = DataSet()
validation_dataset_for_doc.load_dataset_from_df(validation_scorecard_df, schema_filepath=schema_path)


# **Analyze by pipeline steps**
# - Follow each step in `pipeline` and analyze process step by step

# In[ ]:


binner_flag = 0 
cnt = 0 

methods=['percentile', 'range']
step = '0.0_original'
for method in methods: 
    analyzer = Analyzer()
    analyzer.fit(modeling_dataset_for_doc, method=method)
    analyzer.analyze_plot_all({'modeling':modeling_dataset_for_doc}, {'validation': validation_dataset_for_doc}, dirpath='Docs\\{}'.format(card_name), step=step)

for i in pipeline.named_steps: 
    cnt=cnt+1
    transformer = pipeline.named_steps.get(i)
    
    if(i=='binner'): 
        binner_flag = 1
        step="{0:.1f}_{1}".format(cnt-0.5, "binner")
        
        analyzer = Analyzer()
        analyzer.set_binner(transformer)
        analyzer.analyze_plot_all({'modeling':modeling_dataset_for_doc}, {'validation': validation_dataset_for_doc}, dirpath='Docs\\{}'.format(card_name), step=step)

    step = "{0:.1f}_{1}".format(cnt, i)
    modeling_dataset_for_doc = transformer.transform(modeling_dataset_for_doc)
    validation_dataset_for_doc = transformer.transform(validation_dataset_for_doc)

    if(binner_flag==0):
        methods = ['percentile', 'range']
    else: 
        methods = ['original']
        
    for method in methods:
        analyzer = Analyzer()
        analyzer.fit(modeling_dataset_for_doc, method=method)
        analyzer.analyze_plot_all({'modeling':modeling_dataset_for_doc}, {'validation': validation_dataset_for_doc}, dirpath='Docs\\{}'.format(card_name), step=step)
        


# ## Create Scoring Functions

# **SAVE:** save `data_splitter`, `pipeline` (for data preprocess), best performed `modeler` to pkl

# In[ ]:


file_controller = FileController("Models\\DataSplitter.pkl")
file_controller.save(data_splitter)

file_controller = FileController("Models\\{0}\\pipeline.pkl".format(card_name))
file_controller.save(pipeline)

model_id = 'LogistictRegression_140'
modeler = modelers.get(model_id)
file_controller = FileController(filepath="Models\\{0}\\Modeler.pkl".format(card_name))
file_controller.save(modeler)


# **SCORE:** once files are saved to folders `Models\`&`Models\{card_name}\`, `Scorer` class works automatically

# In[ ]:


class Scorer(): 
    def __init__(self, card_name=''): 
        self.card_name = card_name 
        self.schema_path = 'Models\\{0}\\Data_Schema_{0}.csv'.format(card_name)
        
        file_controller = FileController("Models\\DataSplitter.pkl")
        self.data_splitter = file_controller.load()
        
        file_controller = FileController("Models\\{0}\\Pipeline.pkl".format(card_name))
        self.pipeline = file_controller.load()

        file_controller = FileController(filepath="Models\\{0}\\Modeler.pkl".format(card_name))
        self.modeler = file_controller.load()
        
    def score(self, df): 
        df_dict = self.data_splitter.transform(df)
        scorecard_df = df_dict.get(self.card_name)
        
        destination_dataset = DataSet()
        destination_dataset.load_dataset_from_df(scorecard_df, schema_filepath=self.schema_path)
        destination_dataset = self.pipeline.transform(destination_dataset)
        scored_dataset = self.modeler.score(destination_dataset)
    
        return scored_dataset


# In[ ]:


m_df = pd.read_csv('Data\\MODEL_PERSON_T.csv', encoding='big5')

for card_name in ['unsec_both']: 
    scorer = Scorer(card_name = card_name)
    result = scorer.score(m_df)
    
    print(result)


# **VALIDATE:** validate scoring result to ensure correctness in this specific card

# In[ ]:


(result.x!=modeling_dataset.x).sum()

