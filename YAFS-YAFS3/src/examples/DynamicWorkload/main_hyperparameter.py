import csv
import argparse

import itertools
import time
import operator
import copy
import os
import json
from pathlib import Path
import sys
sys.path.insert(0,'/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/')
#print(sys.path)
import networkx as nx
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.*", lineno=872)
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

#from Evolutive_population import Population_Move
from selection_multipleDeploys import  CloudPath_RR
from yafs.path_routing import DeviceSpeedAwareRouting
from compare_avg_delay import compare_avg_delay
from yafs.population import Population
from yafs.distribution import exponential_distribution
from collections import defaultdict
from get_effective_avg_latency import get_effective_avg_latency
import ast

RANDOM_SEED = 1

class MoveStrategy():
    def __init__(self, module_node_mapping, whether_move):
        self.activations = 0
        self.module_node_mapping = module_node_mapping
        self.whether_move=whether_move

    def deploy_module(self,sim,service,node):
        app_name = int(service[0:service.index("_")])
        app = sim.apps[app_name]
        services = app.services
        idDES = sim.deploy_module(app_name, service, services[service], [node])

    def undeploy_module(self,sim,service,node):
        app_name = int(service[0:service.index("_")])
        des = sim.get_DES_from_Service_In_Node(node, app_name, service)
        sim.undeploy_module(app_name, service,des)

    def is_already_deployed(self,sim,service_name,node):
        app_name = service_name[0:service_name.index("_")]

        all_des = []
        for k, v in sim.alloc_DES.items():
            if v == node:
                all_des.append(k)

        for des in sim.alloc_module[int(app_name)][service_name]:
            if des in all_des:
                return True

    def get_current_services(self,sim):
        """ returns a dictionary with name_service and a list of node where they are deployed
        example: defaultdict(<type 'list'>, {u'2_19': [15], u'3_22': [5]})
        """
        # it returns all entities related to a node: modules, sources/users, etc.
        current_services = sim.get_alloc_entities()
        # here, we only need modules (not users)
        current_services = dict((k, v) for k, v in current_services.items() if len(v)>0)
        deployed_services = defaultdict(list)
        for node,services in current_services.items():
            for service_name in services:
                if not "None" in service_name:
                     deployed_services[service_name[service_name.index("#")+1:]].append(node)
        return deployed_services

    def __call__(self, sim, routing):
        routing.invalid_cache_value = True
        current_services = self.get_current_services(sim)
        for service in current_services:
            for currentNode in current_services[service]:
                if service in self.module_node_mapping:
                    newNode = self.module_node_mapping[service]
                    if not self.is_already_deployed(sim,service,newNode):
                        if self.whether_move:
                            self.undeploy_module(sim,service,currentNode)
                            self.deploy_module(sim,service,newNode)
                        else:
                            self.undeploy_module(sim,service,currentNode)
                            self.deploy_module(sim,service,currentNode)
        #print("simulation end time:", sim.env.now)
                        #logging.info("Moved Service %s from %s to %s node"%(service,currentNode,newNode))



class JSONPopulation(Population):
    def __init__(self, json, external_lambda, **kwargs):
        super(JSONPopulation, self).__init__(**kwargs)
        self.data = json
        self.external_lambda = external_lambda
    def initial_allocation(self, sim, app_name):
            #print("self.data",self.data)
            for item in self.data["sources"]:
                if item["app"]== app_name:
                    app_name = item["app"]
                    idtopo = item["id_resource"]
                    #print("item[lambda]:", item['lambda'])
                    #print("self.external_lambda:", self.external_lambda)
                    lambd = self.external_lambda if self.external_lambda else item["lambda"]

                    app = sim.apps[app_name]
                    msg = app.get_message(item["message"])
                    dDistribution = exponential_distribution(name="Exp", lambd=lambd)
                    idsrc = sim.deploy_source(app_name, id_node=idtopo, msg=msg, distribution=dDistribution)



def main_static_net(untiltime, experimentId, degree, depth):

    shift_delay = {}

    alloDe = '/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/examples/DynamicWorkload/data/appExperiment/' + \
        str(experimentId)+'apps_6modules_per_app/allocDefinition.json'
    appDe = '/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/examples/DynamicWorkload/data/appExperiment/' + \
        str(experimentId)+'apps_6modules_per_app/appDefinition.json'
    userDe = '/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/examples/DynamicWorkload/data/appExperiment/' + \
        str(experimentId)+'apps_6modules_per_app/usersDefinition.json'

    folder_results = Path("src/examples/DynamicWorkload/results")
    folder_results.mkdir(parents=True, exist_ok=True)
    folder_results = str(folder_results)+"/"

    t = Topology()
    # In NX-lib there are a lot of Graphs generators
    t.G = nx.generators.balanced_tree(degree, depth)
    attPR_BW = {x: random.randint(1, 10) for x in t.G.edges()}
    nx.set_edge_attributes(t.G, name="PR", values=attPR_BW)
    nx.set_edge_attributes(t.G, name="BW", values=attPR_BW)
    attIPT = {x: 100 for x in t.G.nodes()}
    nx.set_node_attributes(t.G, name="IPT", values=attIPT)
    attRAM = {x: random.randint(1000, 15000) for x in t.G.nodes()}
    # set cloud node size high
    attRAM[0] = 10000
    nx.set_node_attributes(t.G, name="RAM", values=attRAM)
    nodes = t.G.nodes()
    links = t.G.edges
    nodes_ram = attRAM

    """
    APPLICATION
    """
    dataApp = json.load(open(appDe))
    apps = create_applications_from_json(dataApp)

    # store the message bytes from sender to each app's first node
    sender_msg_size = []
    for obj in dataApp:
        for message in obj.get('message', []):
            if message['name'].startswith('M'):
                sender_msg_size.append(message['bytes'])

    """
    PLACEMENT
    """
    placementJson = json.load(open(alloDe))

    """
    SELECTOR algorithm
    """
    selectorPath = DeviceSpeedAwareRouting()

    """
    Deploy Users
    """
    dataPopulation = json.load(open(userDe))
    popu = JSONPopulation(
        name="Statical", json=dataPopulation, external_lambda=None)

    # prepare the data structure for GA.py
    app_list = []
    for i in apps:
        app_list.append(i)
    modules = []
    modules_ram = []
    msg = []
    msg_name_size = {}
    for i in dataApp:
        app_module = i["module"]
        app_msgs = i["message"]
        mini_m = []
        mini_module_ram = []
        mini_msg = []
        for m in app_module:
            mini_m.append(m["name"])
            mini_module_ram.append(m["RAM"])
        for m in app_msgs:
            mini_msg.append([m["s"], m["d"], m["bytes"]])
            msg_name_size[m["name"]] = m["bytes"]
        modules.append(mini_m)
        modules_ram.append(mini_module_ram)
        msg.append((mini_msg))

    # initial plan start with EdgePlacement
    edge_placement = EdgePlacement()
    EP_placement = edge_placement.run(
        nodes, app_list, modules, links, nodes_ram, modules_ram, msg, nx, t.G, attPR_BW)
    #print("initial placement:", EP_placement)

    gen_size = 50  # set generation size for GA.py
    population = 10  # set population size for GA.py
    updated_monitored_latency_edge = {}  # keep each edge's latesed measured latency
    old_plan_measured_latency_workflow = {}

    population = [10, 25, 50, 75]
    generation = [10, 25, 50, 75]

    

    dict_time={}
    dict_measured_latency={}
    fitness={}
    
    for gen_size in range(2,100):
        print("****************Start Test:", (10, gen_size))
        ga = GA()
        sim_parameters = {"topology": t, "population": popu, "selectorPath": selectorPath, "apps": apps,
                      "folder_results": folder_results, "untiltime": untiltime, "old_placement_json": placementJson}
        start=time.time()
        new_plan, new_plan_fitness = ga.run(nodes, app_list, modules, 10, links, nodes_ram, modules_ram, msg, nx, t.G, attPR_BW, gen_size,
                                        shift_delay, EP_placement, sim_parameters, updated_monitored_latency_edge, old_plan_measured_latency_workflow, untiltime, t, True, sender_msg_size)
        end=time.time()

        fitness[(10,gen_size)]=1/new_plan_fitness
        dict_time[(10,gen_size)]=end-start

    return fitness, dict_time


if __name__ == '__main__':

  
    print("Start Hyperparameter Tunning")

    hyper_results_time=[]
    hyper_resulets_lat=[]
    hyper_results_fit=[]

    fitness, dict_time=main_static_net(4000, 4, 2, 4)

    print("fitness:", fitness)
    print("dict_time:", dict_time)
   


   
    for i in range(0,3):
        print("--------------repeat time:", i)
        dict_time, dict_measured_latency, fitness=main_static_net(4000, 4, 2, 4)
        hyper_results_time.append(dict_time)
        hyper_resulets_lat.append(dict_measured_latency)
        hyper_results_fit.append(fitness)
    
    key_sum_time = defaultdict(int)  # 用于存储每个key的总和
    key_count_time = defaultdict(int)  # 用于存储每个key出现的次数

    key_sum_lat = defaultdict(int)  # 用于存储每个key的总和
    key_count_lat = defaultdict(int)  # 用于存储每个key出现的次数

    key_sum_fitness = defaultdict(int)
    key_count_fitness = defaultdict(int)


    # 计算每个key的总和和出现次数
    for d in hyper_results_fit:
        for key, value in d.items():
            key_sum_fitness[key] += value
            key_count_fitness[key] += 1

    # 计算每个key的平均值
    key_avg_fit = {}
    for key in key_sum_fitness:
        key_avg_fit[key] = key_sum_fitness[key] / key_count_fitness[key]


    # 计算每个key的总和和出现次数
    for d in hyper_results_time:
        for key, value in d.items():
            key_sum_time[key] += value
            key_count_time[key] += 1

    # 计算每个key的平均值
    key_avg_time = {}
    for key in key_sum_time:
        key_avg_time[key] = key_sum_time[key] / key_count_time[key]
    

    # 计算每个key的总和和出现次数
    for d in hyper_resulets_lat:
        for key, value in d.items():
            key_sum_lat[key] += value
            key_count_lat[key] += 1

    # 计算每个key的平均值
    key_avg_lat = {}
    for key in key_sum_lat:
        key_avg_lat[key] = key_sum_lat[key] / key_count_lat[key]

    
    #for key in key_avg_lat:
    #    print(str(key) + "-----running time is:"+str(key_avg_time[key])+"   measured latency is:"+str(key_avg_lat[key]))
    
    for key in key_avg_fit:
         print(str(key) + "-----running time is:"+str(key_avg_time[key])+"   measured latency is:"+str(key_avg_lat[key]))
 











            




