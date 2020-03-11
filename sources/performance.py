# -*- coding: utf-8 -*-


#%%

import numpy as np
import pandas as pd

class Performance(): 
    __feature_name = ''
    __data_name = ''
    __iv = 0
    __ks = 0
    __bins = {}
    
    def __init__(self, feature_name, data_name): 
        self.__feature_name = feature_name
        self.__data_name = data_name
        self.__iv = None
        self.__ks = None  
        self.__bounce_pct = None
        self.__bounce_cnt = None
        self.__r_original = None
        self.__r = None  
        self.__bins = {}
        
    def __str__(self): 
        return "{0} ks(%):{1} iv:{2} bounce_pct(%):{3} bounce_cnt:{4} r_original:{5} r:{6}".format(self.__feature_name, 
                self.__ks,  self.__iv, self.__bounce_pct,  self.__bounce_cnt, self.__r_original,  self.__r)
    
    def update(self, feature_name, iv, ks, bins): 
        self.__feature_name, self.__iv, self.__ks, self.__bins = feature_name, iv, ks, bins
        
    def calculate_update_performance(self, r, bins_result): 
        self.__r_original = r
        self.__bins = self.calculate_bins(bins_result)
        self.__iv, self.__ks, self.__bounce_cnt, self.__bounce_pct,  self.__r = self.__calculate_overall_performance(bins_result)
    
    def create_bin_table(self, bins): 
        destination_df = pd.DataFrame(bins)
        first_item = destination_df.columns[0]
        
        if('upper_bound' in bins.get(first_item)): 
            keep_index = ['upper_bound','min','max','count','event','event_rate(%)','event_rate_std(%)','cnt%(%)','woe']
        else: 
            keep_index = ['expected_fields','fields','count','event','event_rate(%)','cnt%(%)','woe','event_rate_std(%)']
        
        destination_df = destination_df.reindex(keep_index)
        
        return destination_df
    
    def __create_bin_table_transpose(self, bins): 
        destination_df = pd.DataFrame.from_dict(bins, orient='index')
        return destination_df

    def calculate_bins(self, bins_result):
        bin_table = self.__create_bin_table_transpose(bins_result)
        bin_table['cnt%(%)'] = np.round((bin_table['count']/sum(bin_table['count']))*100,2)
        bin_table['event_rate(%)'] = np.round((bin_table['event']/bin_table['count'])*100,2)
        bin_table['event%(%)'] = np.round((bin_table['event']/sum(bin_table['event']))*100,2)
        bin_table['non_event%(%)'] = np.round(((bin_table['count'] - bin_table['event'])/sum(bin_table['count'] - bin_table['event']))*100,2) 
        bin_table['event_rate_std(%)'] = np.round((((1-bin_table['event_rate(%)']/100)*(bin_table['event_rate(%)']/100)/(bin_table['count']-1))**0.5)*100,3)
        bin_table['woe'] = np.round(bin_table.apply(lambda x: np.nan if (x['event%(%)']==0 or x['non_event%(%)']==0) else math.log(x["non_event%(%)"]/x['event%(%)']), axis=1),2)
        bins = bin_table.to_dict('index')
        return bins
    
    def __spilt_nulls(self, bins_result):
        bin_table = self.__create_bin_table_transpose(bins_result)
        bin_table_not_na = bin_table[~bin_table.index.isin(['NaN'])]
        bin_table_na = bin_table[bin_table.index.isin(['NaN'])]
        bins_not_na = bin_table_not_na.to_dict('index') 
        try:
            bins_na = bin_table_na.to_dict('index')
        except:
            bins_na = None               
        return bins_not_na, bins_na
    
    def __calculate_overall_performance(self, bins_result):
        '''
        calculate overall performance with only non_na data
        '''
        bins_not_na, bins_na = self.__spilt_nulls(bins_result)
        bins_not_na = self.calculate_bins(bins_not_na)
        bin_table_not_na = self.__create_bin_table_transpose(bins_not_na)
        bin_table_not_na['IV'] = (bin_table_not_na['non_event%(%)'] - bin_table_not_na['event%(%)'])*bin_table_not_na['woe']
        
        IV = bin_table_not_na['IV'].sum()
        IV = np.round(IV,2)
        
        bin_table_not_na['KS(%)'] = abs(bin_table_not_na['non_event%(%)'].cumsum() - bin_table_not_na['event%(%)'].cumsum())
        KS = np.round(max(bin_table_not_na['KS(%)']), 2)
        
        bounce_cnt, bounce_pct, bounce_positions = self.calculate_bounce(bins_not_na, 'event_rate(%)')
        bounce_pct = np.round(bounce_pct*100,2)
        
        try: 
            r = self.__calculate_r(bin_table_not_na)
        except: 
            r = np.nan
        
        return IV, KS, bounce_cnt, bounce_pct, r
    
    def calculate_nan_positions(self, bins, criteria): 
        nan_positions=[]
        for i in bins.keys(): 
            bin_ = bins.get(i)
            if(np.isnan(bin_[criteria])==True): 
                nan_positions.append(i)
        return nan_positions
    
    def calculate_same_positions(self, bins, criteria): 
        same_positions=[]
        for i in bins.keys(): 
            if(i==0): continue
            else: 
                if(bins.get(i)[criteria]==bins.get(i-1)[criteria]): 
                    same_positions.append(i)
        return same_positions

    def calculate_under_portion_positions(self, bins, min_portion = 1/20): 
        positions=[]
        for i in bins.keys(): 
            if(bins.get(i)['cnt%(%)']<(min_portion*100)): 
                positions.append(i)
        return positions
    
    def calculate_bounce(self, bins_not_na, criteria='event_rate(%)'):
        bounce_cnt = 0
        bounce_pct = 0
        bin_table_not_na = self.__create_bin_table_transpose(bins_not_na)
        
        max_bin_index =  bin_table_not_na.loc[bin_table_not_na[criteria] == max(bin_table_not_na[criteria]), ].index.min()
        min_bin_index =  bin_table_not_na[bin_table_not_na[criteria] == min(bin_table_not_na[criteria])].index.min()
        max_range = max(bin_table_not_na[criteria]) - min(bin_table_not_na[criteria])
        
        br_order = ''  # testify the trend of bad rate
        
        if min_bin_index > max_bin_index: 
            br_order = 'descending'
        elif min_bin_index < max_bin_index: 
            br_order = 'ascending'
        else:  # 2 possible situations: (1)all the bins have same bad rate (2)there's only one bin (3)there's only one bin and one 'NaN' bin
            br_order = 'no order'
        
        bounce_positions = []
        bounce_cnt = 0
        bounce_pct = 0
        if br_order == 'descending':
            for i in range(len(bin_table_not_na)-1):
                if bin_table_not_na.iloc[i+1][criteria] > bin_table_not_na.iloc[i][criteria]:
                    bounce_cnt +=1
                    diff =  bin_table_not_na.iloc[i+1][criteria] - bin_table_not_na.iloc[i][criteria]
                    bounce_pct += abs(diff)/max_range
                    bounce_positions.append(i+1)
        elif br_order == 'ascending':
            for i in range(len(bin_table_not_na)-1):
                if (bin_table_not_na.iloc[i+1][criteria] < bin_table_not_na.iloc[i][criteria]):
                    bounce_cnt +=1
                    diff = bin_table_not_na.iloc[i+1][criteria] - bin_table_not_na.iloc[i][criteria]
                    bounce_pct += abs(diff)/max_range
                    bounce_positions.append(i+1)
        else:
            bounce_cnt = 0
        
        bounce_pct = np.round(bounce_pct,2)
        return bounce_cnt, bounce_pct, bounce_positions
    
    def __calculate_r(self, bin_table_not_na):
        x = np.array([])
        y = np.array([])
        for i in range(len(bin_table_not_na)):
            x_length = bin_table_not_na.loc[i, 'count']
            x_value = bin_table_not_na.index[i]
            x = np.append(x, x_value*np.ones(x_length))
            y1_length = bin_table_not_na.loc[i, 'event']
            y0_length = bin_table_not_na.loc[i, 'count'] - bin_table_not_na.loc[i, 'event']
            y1 = np.ones(y1_length)
            y0 = np.zeros(y0_length)
            y = np.append(y,y1)
            y = np.append(y,y0)
        r = np.corrcoef(x,y)[0,1]
        r = np.round(r,2)
        return r


    @property
    def title(self): 
        str_ = self.feature_name
        str_ += " (r={:.2f})".format(self.r_original)
        str_ += "\nIV={0:.2f}, KS={1:.2f}%, r={2:.2f}, bounce={3:.0f}".format(self.iv, self.ks, self.r, self.bounce_cnt)
        return str_

    @property
    def all_(self): 
        return {'IV':self.__iv, 'KS(%)':self.__ks, 'bounce_cnt':self.__bounce_cnt, 'bounce_pct(%)': self.__bounce_pct,'r_original': self.__r_original, 'r': self.__r, 'bins':self.__bins}

    @property
    def overall_performance(self): 
        return {self.data_name: {'IV':self.__iv, 'KS(%)':self.__ks, 'bounce_cnt':self.__bounce_cnt, 'bounce_pct(%)': self.__bounce_pct,'r_original': self.__r_original, 'r': self.__r}}

    @property
    def overall_performance_df(self): 
        overall_performance = self.overall_performance
        
        destination_df = pd.DataFrame(overall_performance)
        destination_df = destination_df.transpose()
        
        return destination_df
    
    @property
    def data_name(self): 
        return self.__data_name
    
    @property
    def iv(self): 
        return self.__iv

    @property
    def ks(self): 
        return self.__ks
    
    @property
    def bounce_cnt(self): 
        return self.__bounce_cnt    

    @property
    def bounce_pct(self): 
        return self.__bounce_pct 

    @property
    def r_original(self): 
        return self.__r_original    

    @property
    def r(self): 
        return self.__r 

    @property 
    def bin_table(self): 
        return self.create_bin_table(self.__bins)
    @property 
    def bin_table_transpose(self): 
        return self.__create_bin_table_transpose(self.__bins)
    
    @property 
    def bins(self): 
        return self.__bins
    
    @property 
    def feature_name(self): 
        return self.__feature_name
    
    def plot(self):
        plotter = Plotter()
        plotter.plot(self)
        
    def plot_to_pdf(self, filepath): 
        plotter = Plotter()
        plotter.plot_to_pdf(self, filepath)
    
    def get(self, indicator):
        if indicator == 'iv':
            return self.__iv
        if indicator == 'ks':
            return self.__ks
        if indicator == 'bounce_cnt':
            return self.__bounce_cnt
        if indicator == 'bounce_pct':
            return self.__bounce_pct 
        if indicator == 'r_original':
            return self.__r_original
        if indicator == 'r':
            return self.__r    
    
   
#%%
import matplotlib.pyplot as plt
import math

class Plotter(): 
    def __init__(self): 
        pass
    
    def plot(self, performance, min_y_axis = 2.5):
        self.min_y_axis = min_y_axis
        self.feature_name = performance.feature_name
        bin_table = performance.bin_table
        self.title = performance.title
        print(bin_table)
        
        figure = plt.figure(num=None, figsize=(10, 6), facecolor='w', edgecolor='k')
#        figure = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')        
        distribution_chart = figure.add_subplot(211)
        distribution_chart = self._arrange_distribution_chart(distribution_chart, bin_table)
        
        lift_chart = distribution_chart.twinx()
        lift_chart = self._arrange_lift_chart(lift_chart, bin_table)
        
        table_chart = figure.add_subplot(212)
        table_chart.axis('off')
        table_chart = self._arrange_table_chart(table_chart, bin_table)

        plt.show()
        return figure
    
    def plot_to_pdf(self, performance, filepath, min_y_axis=2.5): 
        figure = self.plot(performance)
        self.to_pdf(figure, filepath)
    
    
    def __create_text_table(self, bin_table): 
        destination_text_table = pd.DataFrame()
        if("upper_bound" in bin_table.index): 
            for i in ["upper_bound"]: 
                destination_text_table[i] = bin_table.loc[i,].apply(lambda x: "{:,.2f}".format(x) if isinstance(x, str)==False else x)
                
            for i in ["min", "max"]: 
                destination_text_table[i] = bin_table.loc[i,].apply(lambda x: "{:,.2f}".format(float(x)))
        else: 
            for i in ["expected_fields", "fields"]: 
                destination_text_table[i] = bin_table.loc[i,].apply(lambda x: x)
        
        for i in ["count", "event"]: 
            destination_text_table[i] = bin_table.loc[i,].apply(lambda x: "{:,.0f}".format(float(x)))
        for i in ["cnt%(%)"]: 
            destination_text_table[i] = bin_table.loc[i,].apply(lambda x: "{:.1f}%".format(float(x)))
        for i in ["event_rate(%)", "event_rate_std(%)"]: 
            destination_text_table[i] = bin_table.loc[i,].apply(lambda x: "{:.2f}%".format(float(x)))
        for i in ["woe"]: 
            destination_text_table[i] = bin_table.loc[i,].apply(lambda x: "{:.2f}".format(float(x)) if x!="NaN" else x)
        
        destination_text_table = destination_text_table.rename(columns = {"event_rate(%)":"event_rate", "cnt%(%)":"cnt%","event_rate_std(%)":"event_rate_std"})
        
        destination_text_table = destination_text_table.transpose()
        
        return destination_text_table
    
    def _arrange_table_chart(self, table_chart, bin_table, fontsize=10): 
        text_table = self.__create_text_table(bin_table)
        columns = list(text_table.columns)
        rows = list(text_table.index)
        
        cell_text = []
        for row_index in text_table.index:
            row = text_table.loc[row_index,]
            cell_text.append(["{}".format(x) for x in row])
        
        table_chart = table_chart.table( cellText = cell_text
                                , colLabels = columns 
                                , rowLabels = rows
                                , loc = 'center'
                                , colLoc = 'right'
                                , rowLoc = 'right'
                                , edges = 'horizontal'
                                , fontsize = fontsize
                                )
        table_chart.auto_set_font_size(False)
        table_chart.set_fontsize(fontsize)
        table_chart.scale(1, 1.5)
        plt.tight_layout(rect = [0.05, 0, 1, 1])
        
        return table_chart
    
    def _arrange_distribution_chart(self, distribution_chart, bin_table, fontsize=12):         
        width = 0.5
        for i in bin_table.columns: 
            if(i=='NaN'): 
                distribution_chart.bar(len(bin_table.columns)-1, bin_table.loc['cnt%(%)',i], width, label = 'COUNT%', color='whitesmoke', edgecolor='grey')
            else: 
                distribution_chart.bar(i, bin_table.loc['cnt%(%)',i], width, 
                                       label = 'COUNT%', color='silver')
        
        distribution_chart.get_xaxis().set_visible(False)
        distribution_chart.set_ylabel('cnt%(%)', color='dimgrey', fontsize=fontsize+2)
        distribution_chart.set_ylim(0, 100)
        
        distribution_chart.tick_params(axis='y', colors='dimgrey', which='both', labelsize=fontsize)
        distribution_chart.legend(['COUNT%'], loc=2, fontsize=fontsize)
        
        return distribution_chart
    
    def _arrange_lift_chart(self, lift_chart, bin_table, colors='blue', fontsize=12): 
        last_row_num = len(bin_table.columns)-1
        if("NaN" in bin_table.columns): 
            lift_chart.plot(range(last_row_num),
                            bin_table.loc['event_rate(%)',list(range(last_row_num))], 
                            label = 'event_rate(%)', color=colors, marker='o', linestyle='dashed')
            lift_chart.plot(last_row_num, 
                            bin_table.loc['event_rate(%)',].loc["NaN",], 
                            label = 'event_rate(%)', color='whitesmoke', marker='o', markeredgecolor=colors, linestyle='dashed')
        else: 
            lift_chart.plot(list(bin_table.columns), bin_table.loc['event_rate(%)',], 
                            label = 'event_rate(%)', color=colors, marker='o', linestyle='dashed')
        
        lift_chart.set_ylabel('event_rate(%)', color='blue', fontsize=fontsize)
        
        max_y = 0.5+math.ceil(bin_table.loc['event_rate(%)',].max()/self.min_y_axis)*self.min_y_axis
        lift_chart.set_ylim(0, max_y)
        lift_chart.set_title(self.title, fontsize=fontsize+2)
        
        lift_chart.tick_params(axis='y', colors='blue', which='both', labelsize=fontsize)
        lift_chart.legend(['event_rate(%)'], loc=1, fontsize=fontsize)
        
        return lift_chart

    def to_pdf(self, figure, filepath):
        figure.savefig(filepath, format='pdf')
    
#%%

class MultiDataPerformances(): 
    __feature_name = None
    __modeling_performance = None
    __validation_performances = {}
    
    def __init__(self, feature_name): 
        self.__modeling_performance = None
        self.__validation_performances = {}
        self.__feature_name = feature_name
    
    def calculate_update_modeling_performance(self, r, m_bins_result): 
        self.__modeling_performance = self.calculate_modeling_performance(r, m_bins_result)
        
    def update_modeling_performance(self, modeling_performance): 
        self.__modeling_performance = modeling_performance
        
    def calculate_modeling_performance(self, r, bins_result): 
        modeling_performance = Performance('text', 'modeling')
        modeling_performance.calculate_update_performance(r, bins_result)
        return modeling_performance
    
    def calculate_append_validation(self, r, bins_result, validation_dataset_name): 
        performance = self.calculate_validation_performance(r, bins_result, validation_dataset_name)
        self.append_validation_performance(performance)

    def append_validation_performance(self, validation_performance):        
        validation_dataset_name = validation_performance['performance'].data_name
        if validation_dataset_name in self.__validation_performances.keys():
            raise NameError('The name of the validation dataset has been used.')
        else:
            self.__validation_performances[validation_dataset_name] = validation_performance
    
    def calculate_validation_performance(self, r, bins_result, validation_dataset_name): 
        performance_v = Performance(self.feature_name, validation_dataset_name)        
        performance_v.calculate_update_performance(r, bins_result)
        performance_m = self.__modeling_performance
        psi = self.__calculate_psi(performance_m, performance_v)
        validation_performance = {'psi' : psi, 'performance' : performance_v}
        return validation_performance
    
    def __calculate_psi(self, performance_m, performance_v):
        bin_table_m = performance_m.bin_table_transpose
        bin_table_v = performance_v.bin_table_transpose
        bin_table_m['m_cnt%'] = bin_table_m['count']/(bin_table_m['count'].sum())
        bin_table_v['v_cnt%'] = bin_table_v['count']/(bin_table_v['count'].sum())
        calculation_table = pd.concat([bin_table_m['m_cnt%'], bin_table_v['v_cnt%']], axis = 1)
        calculation_table['psi'] = calculation_table.apply(lambda x: (x['m_cnt%']-x['v_cnt%'])*np.log((x['m_cnt%']/x['v_cnt%'])) if x['v_cnt%']!=0 else np.nan, axis=1)
        
        psi = calculation_table['psi'].sum()
        
        return psi    
    
    def plot_specific_data_to_pdf(self, data, path): 
        if(data=='modeling'): 
            data_performance = self.modeling_performance
        else: 
            data_performance = self.__validation_performances.get(data)['performance']
        plotter = Plotter()
        plotter.plot_to_pdf(data_performance, path)
    
    def plot(self): 
        plotter = MultiDataPlotter()
        plotter.plot(self)

    def plot_to_pdf(self, path): 
        plotter = MultiDataPlotter()
        plotter.plot_to_pdf(self, path)
        
    def plot_specific_data(self, data = 'modeling'): 
        if(data=='modeling'): 
            data_performance = self.modeling_performance
        else: 
            data_performance = self.__validation_performances.get(data)['performance']
        plotter = Plotter()
        plotter.plot(data_performance)

    @property
    def feature_name(self): 
        return self.__feature_name
    
    @property
    def modeling_performance(self): 
        return self.__modeling_performance
    
    @property 
    def validation_list(self): 
        return self.__validation_performances.keys()
    
    def get_validation_performance(self, data):
        data_performance = self.__validation_performances.get(data)['performance']
        return data_performance
    
    @property
    def overall_performance_df(self): 
        destination_df = pd.DataFrame()
        destination_df = self.__modeling_performance.overall_performance_df
        destination_df["PSI"] = "NaN"
        
        for data in self.__validation_performances: 
            performance = self.__validation_performances.get(data)['performance']
            df = performance.overall_performance_df
            df["PSI"] = np.round(self.__validation_performances.get(data)['psi'],2)
            
            destination_df = destination_df.append(df)
        
        return destination_df
    
    @property 
    def bin_tables(self): 
        destination_dict = {}
        destination_dict.update({'modeling':self.__modeling_performance.bin_table})
        
        for data in self.__validation_performances: 
            performance = self.__validation_performances.get(data)['performance']
            destination_dict.update({performance.data_name:performance.bin_table})
        
        return destination_dict
    
    @property
    def lift_chart_base(self): 
        event_rate_df = pd.DataFrame()
        count_pct_df = pd.DataFrame()
        
        for i in self.bin_tables.keys(): 
            overall_performance = self.overall_performance_df.copy(deep=True)
            overall_performance = overall_performance[overall_performance.index==i]
            psi = overall_performance.loc[:,'PSI'].values[0]
            
            bin_table = self.bin_tables.get(i)
            count = bin_table[bin_table.index=='count'].sum(axis=1).values[0]
            
            if(i!='modeling'): 
                event_rate_index = '{0}_Y% ({1:,.0f}; PSI={2:.2f})'.format(i, count, psi)
                count_pct_index = '{0}_cnt% ({1:,.0f}; PSI={2:.2f})'.format(i, count, psi)
            else: 
                event_rate_index = '{0}_Y% ({1:,.0f})'.format(i, count, psi)
                count_pct_index = '{0}_cnt% ({1:,.0f})'.format(i, count, psi)
                
            event_rate = bin_table[bin_table.index=='event_rate(%)'].copy(deep=True)
            event_rate.index = [event_rate_index]
            
            count_pct = bin_table[bin_table.index=='cnt%(%)'].copy(deep=True)
            count_pct.index = [count_pct_index]
            
            event_rate_df = event_rate_df.append(event_rate)
            count_pct_df = count_pct_df.append(count_pct)
        
        destination_df = pd.concat([event_rate_df, count_pct_df], axis=0)
        destination_df = destination_df.apply(lambda x: x/100, axis=1)

        return destination_df
    
    @property
    def table_chart_base(self): 
        destination_df = pd.DataFrame()
        
        for i in self.bin_tables.keys(): 
            bin_table = self.bin_tables.get(i)
            if(i=='modeling'): 
                try: 
                    range_table = bin_table.loc[['upper_bound', 'min', 'max'],:]
                except: 
                    range_table = bin_table.loc[['expected_fields','fields'],:]
            
            bin_table = bin_table.loc[['count','event'],:]
            index_ = bin_table.index.to_series()
            index_ = index_.apply(lambda x: '{0}__{1}'.format(i, x))
            bin_table.index = index_
            
            destination_df = destination_df.append(range_table)
            destination_df = destination_df.append(bin_table)
            range_table = pd.DataFrame()

        return destination_df
    
#%%
from matplotlib import cm 

class MultiDataPlotter(): 
    def __init__(self): 
        pass
    
    def plot(self, multi_data_performances, colormap="CMRmap_r", max_=0.9, min_=0.25, legend_loc=1, fontsize=12): 
        bin_tables = multi_data_performances.bin_tables
        overall_performance_table = multi_data_performances.overall_performance_df
        
        colors = self._get_colors(len(bin_tables), colormap, max_, min_)
        
        figure = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
        
        lift_chart = figure.add_subplot(211)
        lift_chart = self._arrange_lift_chart(lift_chart, multi_data_performances, colors, legend_loc, fontsize)

        distribution_chart = figure.add_subplot(413)
        distribution_chart = self._arrange_distribution_chart(distribution_chart, bin_tables, colors, fontsize)
        
        table_chart = figure.add_subplot(414)
        table_chart.axis('off')
        table_chart = self._arrange_table_chart(table_chart, overall_performance_table, fontsize-2)
        
        plt.show()
        return figure
    
    def _split_nulls(self, bin_table): 
        na_columns = ['NaN']
        not_na_columns = list(bin_table.columns)
        not_na_columns.remove('NaN')
        
        bin_table_na = bin_table[na_columns]
        bin_table_not_na = bin_table[not_na_columns]
        
        return bin_table_na, bin_table_not_na
    
    def plot_to_pdf(self, multi_data_performances, filepath, colormap="CMRmap_r", max_=0.9, min_=0.25, legend_loc=1, fontsize=12): 
        figure = self.plot(multi_data_performances, colormap, max_, min_, legend_loc, fontsize)
        self.to_pdf(figure, filepath)
        
    def to_pdf(self, figure, filepath):
        figure.savefig(filepath, format='pdf')
    
    def __plot_lift_line(self, lift_chart, bin_table, color, linewidth): 
        if('NaN' in list(bin_table.columns)): 
            bin_table_na, bin_table_not_na = self._split_nulls(bin_table)
            
            lift_chart.plot(list(bin_table_not_na.columns), bin_table_not_na.loc['event_rate(%)',], 
                            label = 'event_rate(%)', 
                            color=color, marker='o', linestyle='--', linewidth=linewidth)
            lift_chart.plot(len(bin_table.columns)-1, bin_table_na.loc['event_rate(%)',], 
                            label = 'event_rate(%)', 
                            color='white', marker='o', markeredgecolor=color, linestyle='--', linewidth=linewidth)
        else: 
            lift_chart.plot(list(bin_table.columns), bin_table.loc['event_rate(%)',], 
                            label = 'event_rate(%)', 
                            color=color, marker='o', linestyle='--', linewidth=linewidth)
            
        return lift_chart

    def __get_lines(self, lift_chart, bin_tables): 
        lines=[]
        i=0
        for data in bin_tables.keys(): 
            bin_table = bin_tables.get(data)
            if('NaN' in list(bin_table.columns)): 
                lines.append(lift_chart.lines[i*2])
            else: 
                lines.append(lift_chart.lines[i])
            i+=1
                
        return lines

    def _arrange_lift_chart(self, lift_chart, multi_data_performances, colors, legend_loc=1, fontsize=14, min_y_all=2.5): 
        bin_tables = multi_data_performances.bin_tables
        model_name = multi_data_performances.feature_name
        
        cnt = 0 
        max_y_all = 0
        
        for tag in bin_tables: 
            bin_table = bin_tables.get(tag)
            
            color = colors[cnt]
            cnt = cnt+1
            
            x_ticks = range(len(bin_table.columns))
            
            max_y = 0.5+math.ceil(bin_table.loc['event_rate(%)',].max()/min_y_all)*min_y_all
            max_y_all = max(max_y, max_y_all)
            
            if(tag == 'modeling'): 
                linewidth = 4 
            else: 
                linewidth = 2
                
            lift_chart = self.__plot_lift_line(lift_chart, bin_table, color, linewidth)
        
        lines = self.__get_lines(lift_chart, bin_tables)
        
        lift_chart.set_ylabel('event_rate(%)', color='black', fontsize=fontsize)
        lift_chart.set_ylim(0, max_y_all)
        lift_chart.set_xticks(x_ticks)

        lift_chart.get_xaxis().set_visible(True)
        lift_chart.set_xlim(min(x_ticks)-0.5,max(x_ticks)+0.5)
        lift_chart.set_title("{0}".format(model_name), fontsize=fontsize+2)
        
        lift_chart.tick_params(axis='x', colors='black', which='both', labelsize=fontsize)
        lift_chart.tick_params(axis='y', colors='black', which='both', labelsize=fontsize)
        lift_chart.legend(lines, list(bin_tables.keys()), loc=legend_loc)
        
        return lift_chart
    
    def _arrange_distribution_chart(self, distribution_chart, bin_tables, colors, fontsize, min_y_all=2.5): 
        cnt = 0
        max_y_all = 20
        for data in bin_tables: 
            bin_table = bin_tables.get(data)

            color = colors[cnt]
            cnt = cnt+1
            
            max_y = math.ceil(bin_table.loc['cnt%(%)',].max()/min_y_all)*min_y_all
            max_y_all = max(max_y, max_y_all)


            seg_cnt = len(bin_tables)+2
            half_cnt = len(bin_tables)/2
            width = 1/seg_cnt
            x_ticks = range(len(bin_table.columns))
            x_locations = [x+(cnt-half_cnt)*(width) for x in list(x_ticks)]
            
            for i in bin_table.columns: 
                if(i=='NaN'): 
                    distribution_chart.bar(x_locations[len(x_locations)-1], bin_table.loc['cnt%(%)',i], width, label = 'COUNT%', color='white', edgecolor=color)
                else: 
                    distribution_chart.bar(x_locations[i], bin_table.loc['cnt%(%)',i], width, 
                                           label = 'COUNT%', color=color)
            
            distribution_chart.get_xaxis().set_visible(False)
            distribution_chart.set_xlabel(''.format(data), fontsize=fontsize)
            
            distribution_chart.set_xticks(x_ticks)
            distribution_chart.set_xlim(min(x_ticks)-0.5,max(x_ticks)+0.5)
            distribution_chart.set_ylabel('cnt%(%)', color='black', fontsize=fontsize)
            distribution_chart.set_ylim(0, max_y_all)
            
            distribution_chart.set_xticks(x_ticks)
            
            distribution_chart.tick_params(axis='x', colors='black', which='both', labelsize=fontsize)
            distribution_chart.tick_params(axis='y', colors='black', which='both', labelsize=fontsize)

        return distribution_chart
    
    def _arrange_table_chart(self, table_chart, info_table, fontsize): 
        rows = list(info_table.index)
        columns = info_table.columns

        cell_text = []
        for row_index in info_table.index:
            row = info_table.loc[row_index,]
            cell_text.append(["{}".format(x) for x in row])
        
        table_chart = table_chart.table( cellText = cell_text
                                , colLabels = columns 
                                , rowLabels = rows
                                , loc = 'center'
                                , colLoc = 'right'
                                , rowLoc = 'right'
                                , edges = 'horizontal'
                                , fontsize = fontsize
                                )
        table_chart.auto_set_font_size(False)
        table_chart.set_fontsize(fontsize)
        table_chart.scale(1, 1.5)
        plt.tight_layout(rect = [0.05, 0, 1, 1])
        
        return table_chart
    
    def __display_colorlist(self, color_list): 
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        figure = plt.figure()
        axis = figure.add_subplot(111)
        axis.imshow(gradient,  aspect='auto', cmap = color_list, origin='lower')
        figure.show()
    
    
    def _get_colors(self, dataset_cnt, colormap, max_, min_): 
        
        color_list = cm.get_cmap(colormap, 256)
#        self.__display_colorlist(color_list)

        colors=['darkgray']
        max_ = max_
        min_ = min_
        cnt = dataset_cnt-1
        if(cnt==1): 
            gap=0
        else: 
            gap = (max_-min_)/(cnt-1)
        
        for i in range(cnt): 
            colors.append(color_list(max_-i*gap))
        
        return colors
