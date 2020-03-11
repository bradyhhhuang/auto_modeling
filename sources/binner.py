# -*- coding: utf-8 -*-

#%%
import pandas as pd
import numpy as np 
import copy
import PyPDF2
from PyPDF2 import PdfFileReader
from sources.data_set import DataSet
from sources.performance import Performance, MultiDataPerformances

class __FeatureBinner(): 
    def __init__(self): 
        pass

    def _split_nulls(self, data, y = None):
        feature = data.columns[0]
        data_not_na = data[~data[feature].isna()]
        data_na = data[data[feature].isna()]
        
        if(isinstance(y, pd.DataFrame)): 
            y_not_na = y.loc[data_not_na.index]
            y_na = y.loc[data_na.index]
        else: 
            y_not_na = None 
            y_na = None 
        
        return data_not_na, data_na, y_not_na, y_na
    
    def transform(self, source_data):
        data_not_na, data_na, y_not_na, y_na = self._split_nulls(data=source_data)
        
        destination_data_na = data_na.copy(deep=True)
        
        if(len(data_not_na)!=0): 
            destination_series = data_not_na.apply(self._assign_bin_value, axis=1)
            destination_data_not_na = pd.DataFrame()
            destination_data_not_na[self.feature_name] = destination_series
        else: 
            destination_data_not_na = pd.DataFrame() 
        
        destination_data = pd.concat([destination_data_not_na, destination_data_na], axis=0)
        destination_data = destination_data.sort_index()
        
        return destination_data

    def analyze(self, data, y):
        feature = self.feature_name
        bin_feature = "{}_BIN".format(self.feature_name)
        y_feature = y.columns[0]
        
        bin_data = self.transform(data)
        bin_data.columns = [bin_feature]
        
        data_not_na, data_na, y_not_na, y_na = self._split_nulls(data=data, y=None)
        bin_data_not_na, bin_data_na, y_not_na, y_na = self._split_nulls(data=bin_data, y=y)
        
        all_data_not_na = pd.concat([data_not_na, bin_data_not_na, y_not_na], axis=1)
        groupby_bin_not_na = all_data_not_na.groupby(bin_feature)

        try: 
            corr = np.corrcoef(list(data_not_na[feature]), list(y_not_na[y_feature]))[0,1]
            corr = np.round(corr,2)
        except: 
            corr = np.nan
            
        bin_count_not_na = self.bin_count_not_na
        destination_bins = {}
        for key in range(bin_count_not_na): 
            try: 
                bin_ = groupby_bin_not_na.get_group(key)
            except KeyError: 
                bin_ = None
            destination_bin_result = {key: self._calculate_bin_result_not_na(key, bin_, feature, y_feature)}
            destination_bins.update(destination_bin_result)
            
        if(len(data_na)!=0): 
            all_data_na = pd.concat([data_na, bin_data_na, y_na], axis=1)
            destination_bins.update({"NaN": self._calculate_bin_result_na(all_data_na, feature, y_feature)})
        
        return corr, destination_bins
    
    
    def calculate_multi_data_performances(self, modeling_dataset, validation_datasets): 
        multi_data_performance = MultiDataPerformances(self.feature_name)

        data_name = list(modeling_dataset.keys())[0]
        data, y = modeling_dataset.get(data_name)[0], modeling_dataset.get(data_name)[1]
        m_r, m_bins_result = self.analyze(data, y)
        multi_data_performance.calculate_update_modeling_performance(m_r, m_bins_result)
        
        for i in validation_datasets.keys(): 
            data_name = i
            data, y = validation_datasets.get(data_name)[0], validation_datasets.get(data_name)[1]
            v_r, v_bins_result = self.analyze(data, y)        
            multi_data_performance.calculate_append_validation(v_r, v_bins_result, data_name)
            
        return multi_data_performance
    
    def plot_mutli_datasets(self, modeling_dataset, validation_datasets): 
        multi_data_performances = self.calculate_multi_data_performances(modeling_dataset, validation_datasets)
        multi_data_performances.plot()

    def plot_mutli_datasets_to_pdf(self, modeling_dataset, validation_datasets, filepath): 
        multi_data_performances = self.calculate_multi_data_performances(modeling_dataset, validation_datasets)
        multi_data_performances.plot_to_pdf(filepath)
    
    def calculate_performance(self, dataset): 
        data_name = list(dataset.keys())[0]
        data, y = dataset.get(data_name)[0], dataset.get(data_name)[1]
        feature_name = self.feature_name
        
        r, bins_result = self.analyze(data=data, y=y)
        
        performance = Performance(feature_name, data_name)
        performance.calculate_update_performance(r, bins_result)
        
        return performance
    
    def plot(self, dataset): 
        performance = self.calculate_performance(dataset)
        performance.plot()

    def plot_to_pdf(self, dataset, filepath): 
        performance = self.calculate_performance(dataset)
        performance.plot_to_pdf(filepath)

    def _calculate_bin_result_na(self, bin_, feature_name, y_feature_name): 
        count = bin_[feature_name].isnull().sum()
        y = bin_[y_feature_name].agg('sum')
        
        destination_bin_result = {'count':count, 'event': y}
        return destination_bin_result
    
    
#%%
class NominalFeatureBinner(__FeatureBinner): 
    feature_name = ''
    method = ''
    fields = {0: 'else'}
    
    def __init__(self): 
        self.feature_name, self.method = '', ''
        self.fields={0: 'else'}

    def __str__(self): 
        return "{0}: {1}\n".format(self.feature_name, self.fields)
    
    def fit(self, data, y, method='order', criteria='event_rate(%)', ascending=False): 
        self.feature_name = data.columns[0]
        data_not_na, data_na, y_not_na, y_na = self._split_nulls(data, y)
        
        if(method=='order'): 
            self.fields = self.__create_fields_by_criteria_order(data_not_na, y_not_na, criteria, ascending)
        else: 
            print("invalid input")
            self.upper_bounds = None
    
    def __create_fields_by_criteria_order(self, data, y, criteria, ascending): 
        fields = {}
        bin_data = data.copy(deep=True)
        bin_data.columns = ["{}_BIN".format(self.feature_name)]
        bins_result = self.create_bins(data, bin_data, y)
        
        performance = Performance(self.feature_name, 'modeling')
        performance.calculate_update_performance(np.nan, bins_result)
        bin_table = performance.bin_table.copy(deep=True)
        bin_table = bin_table.sort_values(by=criteria, axis=1, ascending=ascending)
        bin_table.columns = range(len(bin_table.columns))
        
        destination_bin_table = pd.Series(data=bin_table.loc['expected_fields',])
        destination_bin_table.name = 'fields'
        fields = destination_bin_table.to_dict()
        
        return fields
        
    def create_bins(self, original_data, bin_data, y): 
        feature = original_data.columns[0]
        bin_feature = bin_data.columns[0]
        y_feature = y.columns[0]
        
        all_data = pd.concat([original_data, bin_data, y], axis=1)
        groupby_bin = all_data.groupby(bin_feature)
        
        bin_fields = all_data[bin_feature].unique()
        destination_bins = {}
        for key in bin_fields: 
            try: 
                bin_ = groupby_bin.get_group(key)
            except KeyError: 
                bin_ = None
            destination_bin_result = {key: self._calculate_bin_result_only([key], key, bin_, feature, y_feature)}
            destination_bins.update(destination_bin_result)
            
        return  destination_bins
    
    def fit_manual(self, feature_name, fields): 
        self.feature_name = feature_name
        self.method = 'manual_nominal'
        self.fields = fields

    def _assign_bin_value(self, value):
        value = value[self.feature_name]
        fields = self.fields
        bin_ = -1

        for i in fields.keys(): 
            field = fields[i]
            if(field == 'else'): 
                bin_ = i 
                break;
            elif(value in field): 
                bin_ = i 
                break;
        
        return bin_
    
    @property
    def bin_count_not_na(self): 
        return len(self.fields)
    
    def _calculate_bin_result_only(self, expected_fields, bin_num, bin_, feature_name, y_feature_name): 
        if(bin_ is None): 
            count = 0 
            y = 0 
            fields = np.nan
        else: 
            count = bin_[feature_name].count()
            y = bin_[y_feature_name].agg('sum')
            fields = bin_[feature_name].unique()
        
        destination_bin_result = {'count':count, 'event': y, 'expected_fields': expected_fields, 'fields':fields}
        return destination_bin_result
    
    
    def _calculate_bin_result_not_na(self, bin_num, bin_, feature_name, y_feature_name): 
        expected_fields = self.fields[bin_num]
        destination_bin_result = self._calculate_bin_result_only(expected_fields, bin_num, bin_, feature_name, y_feature_name)
        return destination_bin_result
    

#%%
class NumericFeatureBinner(__FeatureBinner):
    feature_name = ''
    method = ''
    upper_bounds = {0: 'else'}
    
    def __init__(self): 
        self.feature_name, self.method = '', ''
        self.upper_bounds = {0: 'else'}

    def __str__(self): 
        return "{0}: {1}\n".format(self.feature_name, self.upper_bounds)
    
    def fit(self, data, y, max_bins, method): 
        self.feature_name = data.columns[0]
        data_not_na, data_na, y_not_na, y_na = self._split_nulls(data, y)
        
        if(method == 'range'): 
            upper_bounds = self.__create_range_upper_bounds(data_not_na, max_bins)
        elif(method=='percentile'): 
            upper_bounds = self.__create_percentile_upper_bounds(data_not_na, max_bins)
        elif(method=='tree'): 
            upper_bounds = self.__create_tree_upper_bounds(data_not_na, y_not_na, max_bins, minimum_size=1/20)
        elif(method=='original'): 
            upper_bounds = self.__create_original_upper_bounds(data_not_na)
        else: 
            print("invalid input")
            upper_bounds = None
            
        self.upper_bounds = self.__cleanse_upper_bounds(upper_bounds)

    
    
    def fit_auto_decrease_maxbins(self, data, y, max_bins, method='tree', criteria='event_rate(%)'): 
        self.feature_name = data.columns[0]
        data_not_na, data_na, y_not_na, y_na = self._split_nulls(data, y)
        
        for i in range(max_bins, 1, -1): 
            self.fit(data_not_na, y_not_na, max_bins=i, method=method)
            r, bins_result_not_na = self.analyze(data_not_na, y_not_na)
            performance = Performance(self.feature_name, 'modeling')
            performance.calculate_update_performance(r, bins_result_not_na)
            bounce_cnt, bounce_pct, bounce_positions = performance.calculate_bounce(performance.bins, criteria)
            nan_positions = performance.calculate_nan_positions(performance.bins, criteria)
            same_positions = performance.calculate_same_positions(performance.bins, criteria)
            
            if(bounce_cnt+len(nan_positions)+len(same_positions)==0): break


    def fit_auto_merge_bins(self, data, y, max_bins, method='range', criteria='woe'): 
        self.feature_name = data.columns[0]
        data_not_na, data_na, y_not_na, y_na = self._split_nulls(data, y)
        
        self.fit(data_not_na, y_not_na, max_bins=max_bins, method=method)
        r, bins_result_not_na = self.analyze(data_not_na, y_not_na)
        performance = Performance(self.feature_name, 'modeling')
        performance.calculate_update_performance(r, bins_result_not_na)
#        performance.plot()
        
        bounce_cnt, bounce_pct, bounce_positions = performance.calculate_bounce(performance.bins, criteria)
        nan_positions = performance.calculate_nan_positions(performance.bins, criteria)
        same_positions = performance.calculate_same_positions(performance.bins, criteria)
        
#        cnt = 0 
        while(bounce_cnt+len(nan_positions)+len(same_positions)!=0): 
            merge_positions = copy.deepcopy(bounce_positions)
            
            merge_positions.extend(nan_positions)
            merge_positions.extend(same_positions)
            merge_positions.sort(reverse=True)
            
            for i in merge_positions: 
                if(i!=0): 
                    self.upper_bounds.update({i-1: self.upper_bounds.get(i)})
                else: 
                    self.upper_bounds.update({i: self.upper_bounds.get(i+1)})

            self.upper_bounds = self.__cleanse_upper_bounds(self.upper_bounds)
            
            r, bins_result_not_na = self.analyze(data_not_na, y_not_na)
            performance = Performance(self.feature_name, 'modeling')
            performance.calculate_update_performance(r, bins_result_not_na)
            bounce_cnt, bounce_pct, bounce_positions = performance.calculate_bounce(performance.bins, criteria)
            nan_positions = performance.calculate_nan_positions(performance.bins, criteria)
            same_positions = performance.calculate_same_positions(performance.bins, criteria)
            
#            print("bounce:", bounce_cnt, "nan:", nan_positions, "same:", same_positions)            
#            performance.plot()
            
            if(len(self.upper_bounds.keys())==1): 
#                cnt+=1
                break;
            
#            if(cnt>1): 
#                raise ValueError("error at {}".format(self.feature_name))
#                break; 
        
    
    def __cleanse_upper_bounds(self, upper_bounds):         
        destination_upper_bounds = dict(upper_bounds)
        
        new_index = []
        cnt=0
        for i in upper_bounds.keys(): 
            destination_upper_bounds = dict(destination_upper_bounds)
            if(i==0): 
                new_index.append(cnt)
                cnt += 1
            elif(upper_bounds.get(i)==upper_bounds.get(i-1)): 
                destination_upper_bounds.pop(i)
            else: 
                new_index.append(cnt)
                cnt += 1
        
        bin_table_transpose = pd.DataFrame.from_dict(destination_upper_bounds, orient='index')
        bin_table_transpose.index = new_index
        bin_table_transpose.columns = ['col']
        bin_table = pd.Series(data=bin_table_transpose.col)
        destination_upper_bounds = bin_table.to_dict()
        
        return destination_upper_bounds
    
    @property
    def bin_count_not_na(self): 
        return len(self.upper_bounds)

    def fit_manual(self, feature_name, upper_bounds): 
        self.feature_name = feature_name
        self.method = 'manual_numeric'
        self.upper_bounds = upper_bounds
    
    def _assign_bin_value(self, value):
        value = value[self.feature_name]
        upper_bounds = self.upper_bounds
        
        bin_ = -1

        for i in upper_bounds.keys(): 
            upper_bound = upper_bounds[i]
            if(upper_bound == 'else'): 
                bin_ = i 
                break;
            elif(value <= upper_bound): 
                bin_ = i 
                break;
        
        return bin_

    def __create_range_upper_bounds(self, data, max_bins): 
        upper_bounds = {}
        feature = data.columns[0]
        
        max_ = data[feature].max()
        min_ = data[feature].min()
        range_ = max_ - min_ 
        
        if(range_==0): 
            upper_bounds = {0: 'else'}
        else: 
            segment_width = range_/max_bins
            for i in range(max_bins): 
                if(i==(max_bins-1)): 
                    upper_bound = 'else'
                else: 
                    upper_bound = min_+segment_width*(i+1)
                upper_bounds.update({i:upper_bound})
       
        return upper_bounds
    
    def __create_original_upper_bounds(self, data): 
        upper_bounds = {}
        feature = data.columns[0]
        values = data[feature].unique()
        values = list(values)
        values.sort(reverse=False)
        
        for i in values: 
            upper_bounds.update({i: i})
        return upper_bounds

    def __create_percentile_upper_bounds(self, data, max_bins):
        import numpy as np
        upper_bounds = {}
        feature = data.columns[0]
        
        for i in range(max_bins):
            if max_bins-1 == i:
                upper_bound = 'else'
            else:
                upper_bound = np.percentile(data[feature], (i+1)/max_bins*100, interpolation = 'midpoint')
            upper_bounds.update({i:upper_bound})
        return upper_bounds
        
    def __create_tree_upper_bounds(self, data, y, max_bins, minimum_size):
        from sklearn import tree
        upper_bounds = {}
        feature = data.columns[0]
        clf = tree.DecisionTreeClassifier(
                                           max_leaf_nodes = max_bins
                                          , min_samples_leaf = minimum_size
                                          , class_weight='balanced'
                                          , random_state=0
                                          )
        
        df = data[[feature]]
        clf.fit(df, y)    
        leave_id = clf.apply(df)
        seg_not_nulls = pd.DataFrame(leave_id, columns=['leaf'])
        seg_not_nulls.index = df.index
        seg_not_nulls = pd.concat([seg_not_nulls, df], axis=1)
        maximum_table = seg_not_nulls.groupby(['leaf']).max()
        maximum_list = maximum_table.iloc[:,0].sort_values(ascending  = True)
        
        bin_count = len(maximum_list)
        for i in (range(bin_count)):
            if bin_count-1== i:
                upper_bound = 'else'
            else:
                upper_bound = maximum_list.iloc[i]
            upper_bounds.update({i:upper_bound})
        return upper_bounds

    def _calculate_bin_result_not_na(self, bin_num, bin_, feature_name, y_feature_name): 
        expected_upper_bound = self.upper_bounds[bin_num]
        destination_bin_result = self._calculate_bin_result_only(expected_upper_bound, bin_, feature_name, y_feature_name)
        return destination_bin_result

        
    def _calculate_bin_result_only(self, expected_upper_bound, bin_, feature_name, y_feature_name): 
        expected_upper_bound = expected_upper_bound
        if(bin_ is None): 
            count = 0 
            y = 0 
            min_ = np.nan
            max_ = np.nan
        else: 
            count = bin_[feature_name].count()
            y = bin_[y_feature_name].agg('sum')
            min_ = bin_[feature_name].min()
            max_ = bin_[feature_name].max()
        
        destination_bin_result = {'count':count, 'event': y, 'upper_bound': expected_upper_bound, 'min':min_, 'max':max_}
        return destination_bin_result
    



#%%
class Binner(): 
    _feature_binners = {}
    all_nan_features = []
    
    def __init__(self): 
        self._feature_binners = {}
        self.all_nan_features = []

    def __str__(self): 
        str_ = ""
        for i in self._feature_binners.keys(): 
            feature_binner = self._feature_binners.get(i)
            str_ = str_ + "{}".format(feature_binner)
        return str_

    @property
    def feature_binners(self):
        return self._feature_binners
    
    @property
    def feature_binning_rules(self):
        self.__feature_binning_rules = {}
        for feature in self._feature_binners.keys():
            feature_binner = self._feature_binners[feature]
            if 'NumericFeatureBinner' in str(type(feature_binner)):
                binning_rule = feature_binner.upper_bounds
            if 'NominalFeatureBinner' in str(type(feature_binner)):
                binning_rule = feature_binner.fields        
            self.__feature_binning_rules.update({feature:binning_rule})            
        return self.__feature_binning_rules
       
    
    def fit(self, dataset, fake_y=None, nominal_rules={'method':'order', 'criteria':['event_rate(%)','desc']}, numeric_rules={'max_bins':10, 'method':'percentile', 'auto':'merge', 'criteria':'woe'}):
        """
        fit all nominal/numeric features with assigned methods.
        nominal_rules= {'method':str, 'criteria':dict/list}
        numeric_rules= {'max_bins':int, 'method':str, 'auto':str, 'criteria':str}

        **nominal_rules** 
            Type: ``list`` 
            Content: [method, fields]
            ======= ================================================ 
            method  criteria           
            ======= ================================================
            manual  {0: ['高科技電子業','穩定收入族群'], 1: 'else'}
            order   [event_rate(%)', 'asc'/'desc']
            ======= ================================================
        
        **numeric_rules** 
            Type: ``list`` 
            Content: [auto, max_bins, method, criteria]
            ========= ========= ======================================= 
            auto      max_bins  method         
            ========= ========= =======================================
            none      int       
            decrease  int
            merge     int
            ========= ========= =======================================   
        
        """
        schema = dataset.attribute_list.schema
        schema = schema[schema.Category=='X'][['Field','Type']]
        schema.set_index('Field', inplace = True)
        feature_types = schema.to_dict()['Type']

        for feature in feature_types.keys(): 
            feature_type = feature_types[feature]    
            
            if feature_type == 'nominal':
                features_n_rules = {feature: {'feature_type':feature_type, 'rules':nominal_rules}} 
            
            elif feature_type == 'numeric':
                features_n_rules = {feature: {'feature_type':feature_type, 'rules':numeric_rules}}
                 
            self.fit_specific_features(dataset, **features_n_rules)

    
    def fit_specific_features(self, dataset, **features_n_rules): 
        """
        NOMINAL kwargs = {feature_1: {'feature_type':'nominal', 'rules':{'method':str, 'criteria':str}}, feature_2:{}, feature_3:{}}
        NUMERIC kwargs = {feature_1: {'feature_type':'numeric','rules':{'max_bins':int, 'method':str, 'auto':str, 'criteria':str}}, feature_2:{}, feature_3:{}}

        nominal_rules= {feature1: {'method':str, 'criteria':str}
        numeric_rules= {'max_bins':int, 'method':str, 'auto':str, 'criteria':str}
        
        """
        for feature in features_n_rules.keys():    
            data, y = dataset.x[[feature]], dataset.y
            feature_type = features_n_rules[feature]['feature_type']
            rules = features_n_rules[feature]['rules']
            
            if(data.isnull().sum()[feature]==data.isnull().count())[feature]: 
                print("{} column is all nan, skipped. ".format(feature))
                self.all_nan_features.append(feature)
            
            elif feature_type == 'nominal':
                method = rules['method']
                criteria = rules['criteria']
                
                if method == 'manual':                
                    feature_binner = NominalFeatureBinner()
                    feature_binner.fit_manual(feature, criteria)
                    self._feature_binners.update({feature: feature_binner})
#                    print("\n", feature, ":NOMINAL")
#                    print(feature_binner.fields)  
                
                elif method == 'order':
                    order_criteria = criteria[0]
                    if criteria[1] == 'asc': 
                        order_ascending = True
                    else: 
                        order_ascending = False       
                        
                    feature_binner = NominalFeatureBinner()
                    feature_binner.fit(data=data, y=y, method=method, criteria=order_criteria, ascending=order_ascending)
                    self._feature_binners.update({feature: feature_binner})
#                    print("\n", feature, ":NOMINAL")
#                    print(feature_binner.fields)     

            elif feature_type == 'numeric':  
                max_bins = rules['max_bins']
                method = rules['method']
                auto = rules['auto']
                criteria = rules['criteria']

                if method == 'manual':
                    feature_binner = NumericFeatureBinner()
                    feature_binner.fit_manual(feature, criteria)    
                    self._feature_binners.update({feature: feature_binner})
#                    print("\n", feature, ":NUMERIC")
#                    print(feature_binner.upper_bounds)                
                
                elif method == 'original': 
                    feature_binner = NumericFeatureBinner()
                    feature_binner.fit(data=data, y=y, method=method, max_bins=max_bins)
                    self._feature_binners.update({feature: feature_binner})
                    
                elif auto == 'none':                    
                    feature_binner = NumericFeatureBinner()
                    feature_binner.fit(data=data, y=y, method=method, max_bins=max_bins)
                    self._feature_binners.update({feature: feature_binner})
#                    print("\n", feature, ":NUMERIC")
#                    print(feature_binner.upper_bounds)
                
                elif auto == 'merge':
                    feature_binner = NumericFeatureBinner()
                    feature_binner.fit_auto_merge_bins(data=data, y=y, max_bins=max_bins, method=method, criteria=criteria)
                    self._feature_binners.update({feature: feature_binner})
#                    print("\n", feature, ":NUMERIC")
#                    print(feature_binner.upper_bounds)

                elif auto == 'decrease':
                    feature_binner = NumericFeatureBinner()
                    feature_binner.fit_auto_decrease_maxbins(data=data, y=y, max_bins=max_bins, method=method, criteria=criteria)
                    self._feature_binners.update({feature: feature_binner})
#                    print("\n", feature, ":NUMERIC")
#                    print(feature_binner.upper_bounds)
                

    def remove_specific_features(self, remove_list): 
        for feature in remove_list:
            if feature in self._feature_binners.keys():
                del self._feature_binners[feature]
                print('{} removed from binner'.format(feature))
            else:
                print('{} does not exist in binner'.format(feature))


    def transform(self, dataset):
        destination_x = dataset.x.copy(deep=True)
        destination_y = dataset.y.copy(deep=True)
        destination_attribute_list = copy.deepcopy(dataset.attribute_list)
        
        for feature in self._feature_binners.keys():
            transformed_df = self._feature_binners[feature].transform(dataset.x[[feature]])
            destination_x[feature] = transformed_df

        destination_dataset = DataSet()
        destination_dataset.set_all(destination_x, destination_y, destination_attribute_list)
        return destination_dataset


    def plot(self, complete_dataset, filepath=None, temp_folderpath="Reports\\temp\\" ):
        """
        full_dataset = {'modeling': dataset}
        dataset = {'modeling':[m_data, m_y]}
        """
        plotmerger = PyPDF2.PdfFileMerger()
        
        for feature in self._feature_binners.keys():
            feature_binner = self._feature_binners[feature]
            _index = list(complete_dataset.keys())[0]
            
            dataset_name = list(complete_dataset.keys())[0]
            data_x = complete_dataset[_index].x[[feature]]
            data_y = complete_dataset[_index].y
            
            if filepath != None:
                temp_filepath = temp_folderpath +'{}.pdf'.format(feature)
                feature_binner.plot_to_pdf({dataset_name:[data_x, data_y]}, filepath=temp_filepath)              
                plotmerger.append(temp_filepath)
            else: 
                feature_binner.plot({dataset_name:[data_x, data_y]})

        if filepath != None:
            plotmerger.write(filepath)
            plotmerger.close()    

    
#    def plot_multi_datasets_all(self, complete_modeling_dataset, complete_validation_datasets, filepath=None, temp_folderpath="Reports\\temp2\\" ):
#        """
#        modeling_dataset = {'modeling':modelnig_dataset}
#        validation_datasets = {'ds_1':validation_dataset,'ds_2':validation_dataset}
#        dataset = {'modeling':[m_data, m_y]}
#        """        
#        plotmerger = PyPDF2.PdfFileMerger()
#
#        for feature in self._feature_binners.keys():
#            feature_binner = binner._feature_binners[feature]
#            
#            m_index = list(complete_modeling_dataset.keys())[0]
#            m_dataset_name = list(complete_modeling_dataset.keys())[0]
#            m_data_x = complete_modeling_dataset[m_index].x[[feature]]
#            m_data_y = complete_modeling_dataset[m_index].y
#            
#            modeling_dataset = {m_dataset_name:[m_data_x, m_data_y]}    
#            validation_datasets = {}
#            
#            for dataset in complete_validation_datasets.keys():       
#                data_x = complete_validation_datasets[dataset].x[[feature]]
#                data_y = complete_validation_datasets[dataset].y
#                validation_datasets.update({dataset:[data_x, data_y]})
#                        
#            if filepath != None:
#                temp_filepath = temp_folderpath +'{}.pdf'.format(feature)
#                feature_binner.plot_mutli_datasets_to_pdf(modeling_dataset, validation_datasets, filepath=temp_filepath)
#                plotmerger.append(PdfFileReader(temp_filepath))                
#            else: 
#                feature_binner.plot_mutli_datasets(modeling_dataset, validation_datasets)
#        
#        plotmerger.write(filepath)
#        plotmerger.close() 

    def plot_multi_datasets(self, complete_modeling_dataset, complete_validation_datasets, filepath=None, temp_folderpath="Reports\\temp2\\" ):
        """
        modeling_dataset = {'modeling':modelnig_dataset}
        validation_datasets = {'ds_1':validation_dataset,'ds_2':validation_dataset}
        dataset = {'modeling':[m_data, m_y]}
        """        
        plotwriter = PyPDF2.PdfFileWriter()

        for feature in self._feature_binners.keys():
            feature_binner = self._feature_binners[feature]
            
            m_index = list(complete_modeling_dataset.keys())[0]
            m_dataset_name = list(complete_modeling_dataset.keys())[0]
            m_data_x = complete_modeling_dataset[m_index].x[[feature]]
            m_data_y = complete_modeling_dataset[m_index].y
            
            modeling_dataset = {m_dataset_name:[m_data_x, m_data_y]}    
            validation_datasets = {}
            
            for dataset in complete_validation_datasets.keys():       
                data_x = complete_validation_datasets[dataset].x[[feature]]
                data_y = complete_validation_datasets[dataset].y
                validation_datasets.update({dataset:[data_x, data_y]})
                        
            if filepath != None:
                temp_filepath = temp_folderpath +'{}.pdf'.format(feature)
                feature_binner.plot_mutli_datasets_to_pdf(modeling_dataset, validation_datasets, filepath=temp_filepath)
                plot = PdfFileReader(temp_filepath) 
                plotwriter.addPage(plot.getPage(pageNumber=0))
                
            else: 
                feature_binner.plot_mutli_datasets(modeling_dataset, validation_datasets)
        
        if filepath != None:
            with open(filepath, 'wb') as save:
                plotwriter.write(save)


    def analyze_to_csv(self, dataset, filepath): 
        try: 
            import os 
            os.remove(filepath)
        except: 
            pass
        
        with open(filepath, 'w') as file: 
            for feature in self._feature_binners.keys():
                x = dataset.x[[feature]].copy(deep=True)
                y = dataset.y.copy(deep=True)
                
                feature_binner = self._feature_binners[feature]
                feature_performance = feature_binner.calculate_performance({'modeling':[x,y]})
            
                bin_table = feature_performance.bin_table
                file.write(feature_performance.title)
                file.write('\n')
                bin_table.to_csv(file, index=True, quoting=2, sep=',', line_terminator="\r")
                file.write("\n\n")

                
    def analyze_multi_datasets_to_csv(self, complete_modeling_dataset, complete_validation_datasets, filepath): 
        try: 
            import os 
            os.remove(filepath)
        except: 
            pass
        
        modeling_dataset = complete_modeling_dataset.get('modeling')
        
        with open(filepath, 'w') as file: 
            for feature in self._feature_binners.keys():
                modeling_x = modeling_dataset.x[[feature]].copy(deep=True)
                modeling_y = modeling_dataset.y.copy(deep=True)
                input_modeling_dataset = {'modeling':[modeling_x, modeling_y]}
                
                input_validation_datasets = {}
                for i in complete_validation_datasets.keys(): 
                    validation_dataset = complete_validation_datasets.get(i)
                    x = validation_dataset.x[[feature]].copy(deep=True)
                    y = validation_dataset.y.copy(deep=True)
                    input_validation_datasets.update({i:[x,y]})

                feature_binner = self._feature_binners[feature]
                feature_performance = feature_binner.calculate_multi_data_performances(input_modeling_dataset, input_validation_datasets)
                
                title = feature_performance.feature_name
                lift_chart_base = feature_performance.lift_chart_base
                table_chart_base = feature_performance.table_chart_base
                
                file.write(title)
                file.write('\n'*1)
                lift_chart_base.to_csv(file, index=True, quoting=2, sep=',', line_terminator="\r")
                file.write('\n'*1)
                table_chart_base.to_csv(file, index=True, quoting=2, sep=',', line_terminator="\r")
                file.write('\n'*2)

    

#%%
#        
#import pandas as pd
#from sources.data_set import DataSet
#import numpy as np

#if __name__ =='__main__': 
#    schema_path = 'Data\\Data_Schema.csv'
#    m_filepath = 'Data\\MODEL_PERSON_T.csv'
#    v_filepath = 'Data\\MODEL_PERSON_V.csv'
#    v2_filepath = 'Data\\MODEL_PROD_V_RPL.csv'
#    
#    modeling_dataset = DataSet()
#    modeling_dataset.load_all_from_csv(dataset_filepath=m_filepath, schema_filepath=schema_path)
#    validation_dataset = DataSet()
#    validation_dataset.load_all_from_csv(dataset_filepath=v_filepath, schema_filepath=schema_path)
#    validation_dataset_prod = DataSet()
#    validation_dataset_prod.load_all_from_csv(dataset_filepath=v2_filepath, schema_filepath=schema_path)
#
#    binner = Binner()
#    binner.fit(modeling_dataset, 
#                   nominal_rules={'method':'order', 'criteria':['event_rate(%)','desc']}, 
#                   numeric_rules={'max_bins':10, 'method':'percentile', 'auto':'none', 'criteria':None})    
#
#    binner.fit_specific_features(modeling_dataset, **{'HOL_P12_0009': {'feature_type':'numeric', 
#                                                                       'rules':{'max_bins':10, 'method':'percentile', 'auto':'none', 'criteria':None}},
#                                                      'USE_P12_0779': {'feature_type':'numeric',
#                                                                       'rules':{'max_bins':10, 'method':'tree', 'auto':'decrease', 'criteria':'woe'}},
#                                                      'YEARLY_INCOME': {'feature_type':'numeric',
#                                                                        'rules':{'max_bins':10, 'method':'range', 'auto':'merge', 'criteria':'event_rate(%)'}},
#                                                      'CORP_TYPE_NEW': {'feature_type':'nominal',
#                                                                        'rules':{'method':'manual', 'criteria':{0: ['高科技電子業','穩定收入族群'], 1: 'else'}}},
#                                                      'ADDR_NEW': {'feature_type':'nominal',
#                                                                   'rules':{'method':'order', 'criteria':['event_rate(%)','desc']}}
#                                                      })
#    binner.feature_binning_rules    
#    print(binner)
##
##    binner.remove_specific_features(remove_list = ['HOL_P12_0009', 'HOL_P10_9'])
##
## 
##    binner.plot_all({'modeling':modeling_dataset}, filepath='Models\\Bivariable_Analysis.pdf')                            
##    binner.plot_multi_datasets_all(complete_modeling_dataset={'modeling':modeling_dataset},
##                                   complete_validation_datasets={'validation':validation_dataset, 'v_prod':validation_dataset_prod},
##                                   filepath='Models\\MultiDataset_Bivariable_Analysis.pdf')
##
##    modeling_dataset = binner.transform_all(modeling_dataset)
##    validation_dataset = binner.transform_all(validation_dataset)
##    validation_dataset_prod = binner.transform_all(validation_dataset_prod)
