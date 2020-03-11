# -*- coding: utf-8 -*-


#class __FeatureImputer(): 
#    feature = ''
#    value = None
#    
#    def __init__(self): 
#        pass
#    
#
#class NominalImputer(__FeatureImputer): 
#    feature = ''
#    value = 'NaN'
#    
#    def __init__(self): 
#        self.feature = ''
#        self.value = 'NaN'
#    
#    def __str__(self): 
#        return "{0}: {1}".format(self.feature, self.value)
#        
#    def fit_most_frequent(self, feature_df): 
#        pass
#    
#    def fit_constant(self, feature, value='NaN'): 
#        self.feature = feature
#        self.value = value
#        
#        
#class NumericImputer(__FeatureImputer): 
#    def __init__(self): 
#        self.feature = ''
#        self.value = 0
#    
#    def __str__(self): 
#        return "{0}: {1:,.2f}".format(self.feature, self.value)
#    
#    def fit_median(self, feature_df): 
#        self.feature = feature_df.columns[0]
#        self.value = feature_df[self.feature].median(axis=0)
#
#    def fit_mean(self, feature_df): 
#        self.feature = feature_df.columns[0]
#        self.value = feature_df[self.feature].mean(axis=0)
#
#    def fit_constant(self, feature, value=0): 
#        self.feature = feature
#        self.value = value
        
class Imputer(): 
    nominal_impute_values = {}
    numeric_impute_values = {}
    order_list = []
    
    def __init__(self): 
        self.nominal_impute_values = {}
        self.numeric_impute_values = {}
        self.order_list = []
        
    def __str__(self): 
        str_ = ""
        for i in self.order_list: 
            if i in self.nominal_impute_values.keys(): 
                str_ = str_ + "{0}: '{1}'\n".format(i, self.nominal_impute_values.get(i))
            if i in self.numeric_impute_values.keys(): 
                str_ = str_ + "{0}: {1:,.2f}\n".format(i, self.numeric_impute_values.get(i))
        return str_
    
    def order(self, order_list): 
        if(set(order_list)!=set(self.nominal_impute_values.keys()).union(set(self.numeric_impute_values.keys()))): 
            raise ValueError("Input order_list doesn't match imputers. Please check. ")
        self.order_list = order_list
    
    def fit(self, dataset, numeric_rule='median', numeric_value=0, nominal_rule='most_frequent', nominal_value='NaN'): 
        x_nominal = dataset.x_nominal
        for i in x_nominal.columns: 
            if(nominal_rule == 'most_frequent'):
                feature_df = x_nominal[i]
                value = feature_df.value_counts().idxmax()
                
            elif(nominal_rule == 'constant'): 
                value = nominal_value
            self.nominal_impute_values.update({i:value})
            
        x_numeric = dataset.x_numeric
        for i in x_numeric.columns: 
            feature_df = x_numeric[i]
            if(numeric_rule == 'median'): 
                value = feature_df.median(axis=0)
            elif(numeric_rule == 'mean'): 
                value = feature_df.mean(axis=0)
            elif(numeric_rule == 'constant'): 
                value = numeric_value
            self.numeric_impute_values.update({i:value})
        
        self.order(list(dataset.attribute_list.x_list))

    
    def fit_specific_features(self, numeric_features={}, nominal_features={}):
#        error_numeric_features = []
#        for i in numeric_features.keys(): 
#            if(i not in self.numeric_impute_values.keys()): 
#                error_numeric_features.append(i)
#        error_nominal_features = []
#        for i in nominal_features.keys(): 
#            if(i not in self.nominal_impute_values.keys()): 
#                error_nominal_features.append(i)
#
#        if(len(error_nominal_features)+len(error_numeric_features)!=0): 
#            raise ValueError("specified features doesn't exist in numeric imputer: {0}\nspecified features doesn't exist in nominal imputer:{1}\n Please check. ".format(error_numeric_features, error_nominal_features))
        
        self.numeric_impute_values.update(numeric_features)
        self.nominal_impute_values.update(nominal_features)
        
    def transform(self, dataset, fake_y=None): 
        x = dataset.x
        
        for feature in self.nominal_impute_values.keys():
            value = self.nominal_impute_values.get(feature)
            x[feature].fillna(value, inplace=True)

        for feature in self.numeric_impute_values.keys():
            value = self.numeric_impute_values.get(feature)
            x[feature].fillna(value, inplace=True)
            
        return dataset

            
#%%
#import pandas as pd
#from sources.data_set import DataSet
#df = pd.read_csv("Data\\MODEL_PERSON_T.csv", encoding='big5', index_col='RNUM')
#df['J10'].mean(axis=0)
#
#feature_df = df[['J10']]
#feature_df.isnull().sum()
#
#dataset = DataSet()
#dataset.load_dataset_from_df(df, schema_filepath='Data\\Data_Schema.csv')
#
#imputer = Imputer()
#imputer.fit(dataset)
#imputer.fit_specific_features(numeric_features={'HOL_P10_0351': 0, 'JOB_YEAR': 10}, 
#                              nominal_features={'CORP_TYPE_NEW': '空值'})
#imputer.order(list(dataset.attribute_list.x_list))
#print(imputer)
#
##print(dataset.x.isnull().sum(axis=0))
#
#datset = imputer.transform(dataset)
#print(dataset.x.isnull().sum(axis=0))
