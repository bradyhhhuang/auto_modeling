# -*- coding: utf-8 -*-

from sources.binner import NumericFeatureBinner
from sources.performance import Performance
import numpy as np
import copy

class FeatureLabelTransformer(): 
    feature_name = ''
    mapping_table = {}
    
    def __init__(self): 
        self.mapping_table = {}
    def __str__(self): 
        str_ = dict()
        str_.update({self.feature_name:self.mapping_table})
#        print(str_)
#        print(self.feature_name)
#        print(self.mapping_table)
#        print()
        
        return "({0}:{1})\n".format(self.feature_name, self.mapping_table)
    
    def fit(self, data, y, label='woe'): 
        feature_name = data.columns[0]
        self.feature_name = feature_name
        y_name = y.columns[0]
        bin_name = "{}_BIN".format(feature_name)
        
        feature_binner = NumericFeatureBinner()
        bins_result = {}

        for i in data[feature_name].unique():
            if(np.isnan(i)==False): 
                all_data = data.loc[data[feature_name]==i,]
                all_data[bin_name] = all_data[feature_name]
                all_data[y_name] = y.loc[data[feature_name]==i,]
                
                bin_result = feature_binner._calculate_bin_result_only(np.nan, all_data, feature_name, y_name)
                bins_result.update({i: bin_result})
            else: 
                raise ValueError("error at {}: not support np.nan yet".format(feature_name))
        
        performance = Performance(feature_name, data_name='modeling')
        performance.calculate_update_performance(0, bins_result)        
        bins = performance.bins
        
        mapping_table = {}
        for i in bins.keys(): 
            value = bins.get(i)[label]
            mapping_table.update({i:value})
        
        self.mapping_table = mapping_table
        
        
    def transform(self, data): 
        """
        input: data 
        output: labeled data
        """
        feature_name = data.columns[0]
        contents = data[feature_name].unique()
        
        diff = set(contents)-set(self.mapping_table.keys())
        if(len(diff)!=0): 
            raise ValueError("error at {0}: met undefined value: {1}".format(self.feature_name, diff))
        
        destination_data = data.copy(deep=True)
        destination_data[feature_name] = destination_data[feature_name].apply(lambda x: self.mapping_table.get(x))
        
        return destination_data
            
#%%
class LabelTransformer():  
    _feature_label_transformers = {}
    
    def __init__(self): 
        self._feature_label_transformers = {}
    
    def __str__(self): 
        str_ = "" 
        for i in self._feature_label_transformers: 
            transformer = self._feature_label_transformers.get(i)
            str_ = str_+"{0}".format(transformer)
        return str_
    
    def fit(self, dataset): 
        for feature in dataset.x.columns: 
            x = dataset.x[[feature]].copy(deep=True)
            y = dataset.y.copy(deep=True)
            
            label_transformer = FeatureLabelTransformer()
            label_transformer.fit(x,y)
            self._feature_label_transformers.update({feature:label_transformer})
    
    def transform(self, dataset): 
        destination_dataset = copy.deepcopy(dataset)
        for feature in dataset.x.columns: 
            if(feature not in self._feature_label_transformers.keys()): 
                raise ValueError("error at {}: no such feature in label transformer")
            else: 
                x = dataset.x[[feature]].copy(deep=True)

                feature_labeler = self._feature_label_transformers.get(feature)
                destination_dataset.x[feature] = feature_labeler.transform(x)
        
        return destination_dataset

