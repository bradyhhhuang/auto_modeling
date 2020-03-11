# -*- coding: utf-8 -*-


#%%
import pandas as pd
import copy
from sources.binner import Binner
from sources.data_set import DataSet
from scipy import stats   

class Modeler():
    clf=None

    def __init__(self, model_id):
        self.model_id = model_id
        
        
    def fit(self, clf, modeling_dataset, validation_datasets, rules):
        clf = copy.deepcopy(clf)
        x = modeling_dataset.get('modeling').x
        y = modeling_dataset.get('modeling').y
        self.__feature_list = list(modeling_dataset.get('modeling').attribute_list.x_list)
        self.__clf = clf.fit(x, y)

        # predict
        modeling_prob_datadict, modeling_prob_datalist = self.__set_prob_datasets(modeling_dataset)
        validation_prob_datadict, validation_prob_datalist = self.__set_prob_datasets(validation_datasets)
                
        # bin            
        self.__binner = Binner()
        self.__binner.fit_specific_features(modeling_prob_datadict['modeling'], 
                                             **{'y_proba': {'feature_type':'numeric', 
                                                            'rules': {'method':rules['method'],'max_bins':rules['max_bins'], 'auto':rules['auto'],'criteria':rules['criteria']}}})   
        # performance
        self.__multiple_performance =  self.__binner.feature_binners.get('y_proba').calculate_multi_data_performances(modeling_prob_datalist, validation_prob_datalist)

        # feature information
        correlation, pvalue = self.__calculate_correlation_pvalue(x, y)
        self.__feature_info = self.__gather_feature_info(correlation, pvalue)
        
        

    def fit_binner(self, modeling_dataset, validation_datasets, rules):
        # predict
        modeling_prob_datadict, modeling_prob_datalist = self.__set_prob_datasets(modeling_dataset)
        validation_prob_datadict, validation_prob_datalist = self.__set_prob_datasets(validation_datasets)
                
        # bin            
        self.__binner = Binner()
        self.__binner.fit_specific_features(modeling_prob_datadict['modeling'], 
                                             **{'y_proba': {'feature_type':'numeric', 
                                                            'rules': {'method':rules['method'],'max_bins':rules['max_bins'], 'auto':rules['auto'],'criteria':rules['criteria']}}})   
        # performance
        self.__multiple_performance =  self.__binner.feature_binners.get('y_proba').calculate_multi_data_performances(modeling_prob_datalist, validation_prob_datalist)



    def predict(self, dataset, proba_schema_path = "Data\proba_schema.csv"):
        y_proba = pd.DataFrame(self.__clf.predict_proba(dataset.x), columns = ['non_y_proba','y_proba'])
        y_proba.index = dataset.x.index
        y_proba = pd.concat([dataset.y, y_proba[['y_proba']]], axis=1)    
        
        prob_dataset = DataSet()
        prob_dataset.load_dataset_from_df(y_proba, schema_filepath=proba_schema_path)        
        return prob_dataset


    def __set_prob_datasets(self, dataset_dictionary):        
        prob_datadict = {}
        prob_datalist = {}
        for dataset in dataset_dictionary.keys():       
            prob_dataset = self.predict(dataset_dictionary[dataset])
            prob_datadict.update({dataset: prob_dataset})
            prob_datalist.update({dataset: [prob_dataset.x, prob_dataset.y]})            
        return prob_datadict, prob_datalist
          

    def score(self, dataset):
        prob_dataset = self.predict(dataset)
        score_dataset = self.__binner.transform(prob_dataset) 
        score = score_dataset.x
        return score


    def plot(self, modeling_dataset, validation_datasets, plot_path='Docs'):        
        # predict
        modeling_prob_datadict, modeling_prob_datalist = self.__set_prob_datasets(modeling_dataset)
        validation_prob_datadict, validation_prob_datalist = self.__set_prob_datasets(validation_datasets)        

        # plot
#        self.__binner.plot(modeling_prob_datadict, filepath='{0}\Score_Lift-{1}.pdf'.format(plot_path, self.model_id))
        self.__binner.plot_multi_datasets(modeling_prob_datadict, validation_prob_datadict, filepath='{0}\MultiDataset_Score_Lift-{1}.pdf'.format(plot_path,self.model_id))
                
        for feature in self.__feature_info.index:
            if self.__feature_info.correction[feature] == True:
                print('[!] CORRECTION : ' + feature)


    def __calculate_correlation_pvalue(self, modeling_x, modeling_y):
        correlation = []
        pvalue = []
        for i in self.__feature_list:
            corr, pv = stats.spearmanr(modeling_x[i], modeling_y)
            correlation.append(corr)
            pvalue.append(pv)
        return correlation, pvalue


    def __gather_feature_info(self, correlation, pvalue):
        """
        Only applies to linear models.
        """
        coefficient = self.__clf.coef_    

        feature_info = pd.concat([pd.Series(self.__feature_list, name='variables'), 
                       pd.Series(coefficient[0], name='coefficient'), 
                       pd.Series(correlation, name='correlation'), 
                       pd.Series(pvalue, name='p-value')], axis = 1)        
        feature_info.set_index('variables', inplace = True)
        
        intercept = self.__clf.intercept_
        feature_info0 = pd.DataFrame({'coefficient': intercept}, index=['intercept'])
        feature_info = feature_info.append(feature_info0)
        
        feature_info['correction'] = feature_info.coefficient*feature_info.correlation
        feature_info['correction'] = feature_info['correction'].apply(lambda x:True if x<0 else False)
      
        return feature_info       

    def output_model_documents(self, modeling_dataset, validation_datasets, doc_directory):
        # predict
        modeling_prob_datadict, modeling_prob_datalist = self.__set_prob_datasets(modeling_dataset)
        validation_prob_datadict, validation_prob_datalist = self.__set_prob_datasets(validation_datasets)        
        
        # to_csv
        self.__binner.analyze_multi_datasets_to_csv(modeling_prob_datadict, validation_prob_datadict, filepath='{0}\{1}-bin_table.csv'.format(doc_directory, self.model_id))
        self.__gather_model_information(filepath='{0}\{1}-model_info.csv'.format(doc_directory, self.model_id))
        

    def __gather_model_information(self, filepath):
        try: 
            import os 
            os.remove(filepath)
        except: 
            pass
        
        with open(filepath, 'w') as file: 
            
            file.write(self.model_id)
            file.write("\n\n")
            file.write(self.__cleanse_text(str(self.__clf)))
            file.write("\n\n")
            self.__multiple_performance.overall_performance_df.to_csv(file, index=True, quoting=2, sep=',', line_terminator="\r")
            file.write("\n\n")
            

    def __cleanse_text(self, text): 
        """
        Text cleansing for parameters.
        """
        if(text==None): 
            return None 
        else: 
            import re
#            text = re.sub('[\s\t]*?\n[\s\t]*?','', text)
            text = re.sub(' ','', text)
            text = re.sub(',',', ', text)
            return text


    @property
    def score_binner(self):
        return self.__binner
    
    @property
    def performance(self):
        return self.__multiple_performance
    
    @property
    def performance_df(self):
        return self.__multiple_performance.overall_performance_df

    @property 
    def clf(self):
        return self.__clf
    
    @property
    def feature_info(self):
        return self.__feature_info
    
  
        
#%%
        
class ModelerRanker():
    
    def __init__(self):
        pass
    
    def set_filters(self, **filters):
        self.filters = filters

    def set_orders(self, **orders):
        self.orders = orders


    def apply_filters(self, modeler_dictionary):
        filtered_model_list = []        
        for model_id in modeler_dictionary.keys():
            modeler = modeler_dictionary.get(model_id)
            qualification = True
            
            for ds in self.filters.keys():
                if ds == 'modeling':
                    ds_performance = modeler.performance.modeling_performance
                    for indicator in self.filters[ds].keys():
                        operator = self.filters[ds][indicator][0]
                        limitation = self.filters[ds][indicator][1]
                        value = ds_performance.get(indicator)                    
                        if operator =='>=':
                            check = value>=limitation                        
                        elif operator == '>':
                            check = value>limitation                        
                        elif operator == '=':
                            check = value==limitation                        
                        elif operator == '<':
                            check = value<limitation                        
                        elif operator == '<=':
                            check = value<=limitation
                        qualification = qualification & check                           
                else:
                    ds_performance = modeler.performance.get_validation_performance(ds)
                    for indicator in self.filters[ds].keys():
                        operator = self.filters[ds][indicator][0]
                        limitation = self.filters[ds][indicator][1]
                        value = ds_performance.get(indicator)                    
                        if operator =='>=':
                            check = value>=limitation                        
                        elif operator == '>':
                            check = value>limitation                        
                        elif operator == '=':
                            check = value==limitation                        
                        elif operator == '<':
                            check = value<limitation                        
                        elif operator == '<=':
                            check = value<=limitation
                        qualification = qualification & check
           
            if qualification == True:
                filtered_model_list.append(model_id)                        
    
        return filtered_model_list

                            
        
    def apply_orders(self, modelers, filtered_model_list):
        base_table = pd.DataFrame()
        features_to_sort = []   
        features_order = []
        
        for order in self.orders.keys():
            ds = self.orders.get(order)[0]     
            indicator = self.orders.get(order)[1]
            asc_ = self.orders.get(order)[2]
            if asc_ == 'asc': asc = True
            else: asc = False
            feature_name = ds + '_' + indicator
            
            value_dict = dict()
            
            for model_id in filtered_model_list:
                modeler = modelers.get(model_id)
                if ds == 'modeling':
                    ds_performance = modeler.performance.modeling_performance
                else:
                    ds_performance = modeler.performance.get_validation_performance(ds)    
                value = ds_performance.get(indicator)           
                value_dict.update({model_id:value})
            value_df = pd.DataFrame.from_dict(value_dict, orient='index',columns=[feature_name])               
            base_table = pd.concat([base_table, value_df],axis=1)
            features_to_sort.append(feature_name)
            features_order.append(asc)
        
        ranking = base_table.sort_values(by=features_to_sort , ascending=features_order)  
        ranking = ranking.reset_index()
        ranking.rename(columns={'index':'model_id'}, inplace = True)    
        ranking['rank'] = pd.Series(ranking.index).apply(lambda x: x+1)
        ranking.set_index('model_id', drop=True, inplace = True)
        
        return ranking
    
    
    def rank(self, modelers):
        filtered_model_list = self.apply_filters(modelers)
        self.ranking = self.apply_orders(modelers, filtered_model_list)

        return self.ranking
        
    
    def top_n_models(self, n):
        top_n_models = self.ranking[self.ranking['rank'] <= n].index.tolist()             
        return top_n_models
             
        
        
        
