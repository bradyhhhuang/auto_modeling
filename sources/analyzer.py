# -*- coding: utf-8 -*-


from sources.binner import Binner

class Analyzer(): 
    binner = None 
    method = '' 
    
    def __init__(self): 
        self.binner = None
    
    def set_binner(self, binner): 
        self.binner = binner
        self.method = 'none'
        
    def fit(self, dataset, method='original', max_bins=10): 
        self.binner = Binner()
        self.binner.fit(dataset, 
                        nominal_rules={'method':'order', 'criteria':['event_rate(%)','desc']}, 
                        numeric_rules={'max_bins':max_bins, 'method':method, 'auto':'none', 'criteria':None})
        self.method = method
        
    def analyze_plot_all(self, complete_modeling_dataset, complete_validation_datasets, dirpath='Docs\\', step=''): 
        binner = self.binner
        all_datasets = dict()
        all_datasets.update(complete_modeling_dataset)
        all_datasets.update(complete_validation_datasets)
        
        for i in all_datasets.keys(): 
            dataset = all_datasets.get(i)
            binner.plot({i:dataset}, filepath='{0}\\{1}_{2}_{3}.pdf'.format(dirpath, step, i, self.method))
            binner.analyze_to_csv(dataset=dataset, filepath='{0}\\{1}_{2}_{3}.csv'.format(dirpath, step, i, self.method))
        
        binner.plot_multi_datasets(complete_modeling_dataset, complete_validation_datasets, filepath='{0}\\{1}_{2}_{3}.pdf'.format(dirpath, step, 'ALL', self.method))
        binner.analyze_multi_datasets_to_csv(complete_modeling_dataset, complete_validation_datasets, filepath='{0}\\{1}_{2}_{3}.csv'.format(dirpath, step, 'ALL', self.method))


