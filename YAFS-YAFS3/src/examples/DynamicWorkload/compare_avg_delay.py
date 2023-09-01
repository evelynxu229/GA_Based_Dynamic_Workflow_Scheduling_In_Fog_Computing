import pandas as pd
import numpy as np


def compare_avg_delay(threshold, resultsPath, untiltime, shift_delay):
    df=pd.read_csv(resultsPath+'_link.csv')
    average_shiftime = df.groupby(['src', 'dst'])['shiftime'].mean()
    avg_shiftime_dict=average_shiftime.to_dict()
    shift_delay.update(avg_shiftime_dict)
    print("shift_time:",average_shiftime )
    print("threshold", threshold)
    return all(mean_value < threshold for mean_value in average_shiftime)

    



    