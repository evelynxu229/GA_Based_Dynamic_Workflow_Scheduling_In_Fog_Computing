import csv
import networkx as nx
import pandas as pd
from collections import defaultdict
"""
current-average-latency: 从sim_trace_link中获取所有的latency总和
Planning-average-latency: 已经从GA中获取到了

current_avg_latency
"""


def get_effective_avg_latency(planning_average_latency, execution_time,  adapt_time, results_path, current_plan, G):
    # Current-average-latency: average latency of workflow in current plan (from monitoring) 第一个plan的
    # Planning-average-latency: average latency of workflow in future plan (from GA)
    # Effective-average-latency: t/(t-adapt_time)*planning-average-latency
    # If effective-average-latency<current-average-latency: 
    #   effective_avg_latency=get_effective_avg_latency()

    current_average_latency=0
    df = pd.read_csv(results_path)
    list_of_dfs = []
    current_df = pd.DataFrame()
    reference_row = df.iloc[0]
    row_list = []
    for index, row in df.iterrows():
        if row['src'] == reference_row['src'] and row['dst'] == reference_row['dst'] and row['message'] == reference_row['message']:
            if row_list:
                list_of_dfs.append(pd.DataFrame(row_list))
                row_list = []
            reference_row = row
        row_list.append(row)
    if row_list:
        list_of_dfs.append(pd.DataFrame(row_list))
    list_of_results = []  
    for i, small_df in enumerate(list_of_dfs):
        if i == len(list_of_dfs) - 1:
            break
        small_df['latency_shift_sum'] = small_df['latency'] + \
            small_df['shiftime']
        def transform_message(message):
            if message.startswith('M'):
                return message.split('.')[-1]  
            else:
                return message.split('_')[0]  
        small_df['grouped_message'] = small_df['message'].apply(
            transform_message)
        grouped = small_df.groupby('grouped_message')
        sum_dict = {}
        for name, group in grouped:
            sum_value = group['latency_shift_sum'].sum()
            sum_dict[name] = sum_value
        list_of_results.append(sum_dict)
    merged_dict = defaultdict(list)
    for data in list_of_results:
        for key, value in data.items():
            merged_dict[key].append(value)
    avg_dict = {}
    for key, values in merged_dict.items():
        avg_value = sum(values) / len(values)
        avg_dict[f'M.USER.APP.{key}'] = avg_value  
    for key, values in avg_dict.items():
        current_average_latency+=values

    effective_average_latency = execution_time / \
        (execution_time-adapt_time)*planning_average_latency

    print("planning_average_latency:", planning_average_latency)
    print("current_average_latency:", current_average_latency)
    print("effective_average_latency:", effective_average_latency)

    if effective_average_latency < current_average_latency:
        return True
    else:
        return False
