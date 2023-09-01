#为了将新的placement进行部署，然后运行这个placement，然后得到shift_time。

#为了实现这个功能，需要实现一个功能，就是将新的placement生成到一个json中，然后让这个simulator读取这个json进行部署，然后测试运行。

#然后生成一个新的csv，再读取，然后更新这个这个csv，更新shift_delay



import argparse

import itertools
import time
import operator
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,'/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/')
print(sys.path)
import networkx as nx
from yafs.application import create_applications_from_json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from yafs.core import Sim
from yafs.application import Application,Message
from yafs.topology import Topology
from yafs.placement import JSONPlacement
from yafs.EdgePlacement import *
from yafs.distribution import *
from yafs.GA import GA

from Evolutive_population import Population_Move
from selection_multipleDeploys import  CloudPath_RR
from yafs.path_routing import DeviceSpeedAwareRouting
from compare_avg_delay import compare_avg_delay
#from compare_latency import adapte_if_exceeds_threshold
from yafs.population import Population
from yafs.distribution import exponentialDistribution
from collections import defaultdict

class JSONPopulation(Population):
    def __init__(self, json, **kwargs):
        super(JSONPopulation, self).__init__(**kwargs)
        self.data = json

    def initial_allocation(self, sim, app_name):
            for item in self.data["sources"]:
                if item["app"]== app_name:
                    app_name = item["app"]
                    idtopo = item["id_resource"]
                    lambd = item["lambda"]
                    app = sim.apps[app_name]
                    msg = app.get_message(item["message"])

                    dDistribution = exponentialDistribution(name="Exp", lambd=lambd)

                    idsrc = sim.deploy_source(app_name, id_node=idtopo, msg=msg, distribution=dDistribution)

"""
    PLACEMENT 
    placementJson = json.load(open(alloDe))
    placement = JSONPlacement(name="Placement", json=placementJson)
"""




class new_simulator():
    def __init__(self,t, population, selectorPath, apps, folder_results,new_palcement_list, untiltime, old_placement_json, shift_delay):
        self.t=t
        self.pop=population
        self.selectorPath=selectorPath
        self.apps=apps
        self.folder_results=folder_results
        self.untiltime=untiltime
        #传进来的新的placement的list
        self.new_placement_list=new_palcement_list
        #根据传进来的list生成的新的placement的json
        self.new_placement_json=None
        #之前的placement，是一个json格式
        self.old_placement_json=old_placement_json
        self.shift_delay=shift_delay
    
    """
    该函数的作用是将list呈现的placement重新放到json中，并且生成新的json，让sim读取成placement
    """
    def deploy_new_placement(self):
        old_json_placement =  self.old_placement_json
        # 将 JSON 数据解码为一个字典
        data = json.loads(old_json_placement)
        for allocation in data["initialAllocation"]:
            app = allocation["app"]
            module_num = int(allocation["module_name"].split("_")[1]) - 1
            allocation["id_resource"] = self.new_placement_list[app][module_num]
        # 将修改后的数据编码回 JSON 格式
        json_data = json.dumps(data, indent=2)
        return json_data
        
    def simulator_run(self):
        self.new_placement_json=self.deploy_new_placement()
        s = Sim(self.t, default_results_path=self.folder_results+"sim_trace_temp")
        for aName in self.apps.keys():
            #print("Deploying app: ",aName)
            pop_app = JSONPopulation(name="Statical_%s"%aName,json={})
            data = []
            for element in self.pop.data["sources"]:
                if element['app'] == aName:
                    data.append(element)
                pop_app.data["sources"]=data
            s.deploy_app2(self.apps[aName], self.new_placement_json, pop_app, self.selectorPath)
        s.run(until=self.untiltime)
    
    def update_shift_delay(self):
        df=pd.read_csv(self.folder_results+'sim_trace_temp_link.csv')
        average_shiftime = df.groupby(['src', 'dst'])['shiftime'].mean()
        avg_shiftime_dict=average_shiftime.to_dict()
        self.shift_delay.update(avg_shiftime_dict)
        return self.shift_delay
        




