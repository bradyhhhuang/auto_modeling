# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:44:31 2019

@author: Heidi
"""

import pandas as pd 
import numpy as np
from sources.binner import NumericFeatureBinner, NominalFeatureBinner

class DataSplitter():
    """According to the assigned segment column, split dataframe into multiple dataframes and store in a dictionary."""

    def __init__(self):
        pass        

    def fit_numeric_rules(self, segment_feature, criteria):
        self.feature_binner = NumericFeatureBinner()
        self.feature_binner.fit_manual(segment_feature, criteria)
        self.scorecard_list = list(criteria.keys())
#        print(self.feature_binner.upper_bounds) 
        """
        criteria = {'low': 0,
                    'medium': 500000,
                    'high':'else'
                    }      
        """
    
    def fit_nominal_rules(self, segment_feature, criteria):
        self.feature_binner = NominalFeatureBinner()
        self.feature_binner.fit_manual(segment_feature, criteria)
        self.scorecard_list = list(criteria.keys())
#        print(self.feature_binner.fields) 
        """
        criteria = {'unsec_both':['UNSEC-BOTH','UNSEC-BOTH-SEC'],
                    'cc':['CC_TXN','UNSEC-CC'],
                    'no_cc_ln':'NO_CC_LN',
                    'unsec':'else'
                    }      
        """
        
    def transform(self, data):
        scorecard_df = self.feature_binner.transform(data[[self.feature_binner.feature_name]])
        data['scorecard']  = scorecard_df         
        df_dict = {}        
        for card in data['scorecard'].unique():
            split_data = data[data.scorecard == card]           
            df_dict[card] = split_data
        return df_dict
    

##1    
#data_splitter = DataSplitter()
#data_splitter.fit_numeric_rules('HOL_P09_0240', {'low': 0, 'medium': 500000,'high':'else'})
#df_dict = data_splitter.transform(m_data)
#
#
##2
#data_splitter = DataSplitter()
#data_splitter.fit_nominal_rules('L1_PATH_NEW', {'unsec_both':['UNSEC-BOTH','UNSEC-BOTH-SEC'],
#                                            'cc':['CC_TXN','UNSEC-CC'],
#                                            'no_cc_ln':'NO_CC_LN',
#                                            'unsec':'else'
#                                            })
#df_dict = data_splitter.transform(m_data)            
