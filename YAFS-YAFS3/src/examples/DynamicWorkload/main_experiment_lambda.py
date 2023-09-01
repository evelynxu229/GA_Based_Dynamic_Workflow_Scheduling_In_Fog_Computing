
import csv
import argparse

import itertools
import time
import operator
import copy
import json
import os
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


RANDOM_SEED = 1


class JSONPopulation(Population):
    def __init__(self, json, external_lambda, **kwargs):
        super(JSONPopulation, self).__init__(**kwargs)
        self.data = json
        self.external_lambda = external_lambda

    def initial_allocation(self, sim, app_name):
        # print("self.data",self.data)
        for item in self.data["sources"]:
            if item["app"] == app_name:
                app_name = item["app"]
                idtopo = item["id_resource"]
                #print("item[lambda]:", item['lambda'])
                #print("self.external_lambda:", self.external_lambda)
                lambd = self.external_lambda if self.external_lambda else item["lambda"]

                app = sim.apps[app_name]
                msg = app.get_message(item["message"])
                dDistribution = exponential_distribution(
                    name="Exp", lambd=lambd)
                idsrc = sim.deploy_source(
                    app_name, id_node=idtopo, msg=msg, distribution=dDistribution)


def main_static_lambda(untiltime, experimentId, degree, depth, lambda_rate):

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
    attRAM = {x: random.randint(200, 7000) for x in t.G.nodes()}
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
    pop = JSONPopulation(
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

    ga = GA()
    sim_parameters = {"topology": t, "population": pop, "selectorPath": selectorPath, "apps": apps,
                      "folder_results": folder_results, "untiltime": untiltime, "old_placement_json": placementJson}
    new_plan, new_plan_fitness = ga.run(nodes, app_list, modules, population, links, nodes_ram, modules_ram, msg, nx, t.G, attPR_BW, gen_size,
                                        shift_delay, EP_placement, sim_parameters, updated_monitored_latency_edge, old_plan_measured_latency_workflow, untiltime, t, True, sender_msg_size)

    """
    next part is about experiment of lambda
    """

    # transfer placement from list into JSONplacement for YAFS simulator
    for allocation in placementJson["initialAllocation"]:
        app = allocation["app"]
        module_num = int(allocation["module_name"].split("_")[1]) - 1
        allocation["id_resource"] = new_plan[app][module_num]
    json_data = json.dumps(placementJson, indent=2)
    placementJson = json.loads(json_data)
    placement = JSONPlacement(name="Placement", json=placementJson)

    s = Sim(t, default_results_path=folder_results + "sim_trace_static")  # create the simuator

    # set up the simulator, use our static lambda
    for aName in apps.keys():
        pop_app = JSONPopulation(name="Statical_%s" %
                                 aName, json={}, external_lambda=lambda_rate)
        data = []
        for element in pop.data["sources"]:
            if element['app'] == aName:
                element['lambda'] = lambda_rate
                data.append(element)
            pop_app.data["sources"] = data
        s.deploy_app2(apps[aName], placement, pop_app, selectorPath)

    # run the simulator
    s.run(until=untiltime)

    

    # collect the measured delay in this rate
    df = pd.read_csv(folder_results+'sim_trace_static_link.csv')
    list_of_dfs = []
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
    list_of_latency_results = []
    list_of_latency_shift_results = []
    for small_df in list_of_dfs:
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
        latency_dict = {}
        latency_shift_dict = {}
        for name, group in grouped:
            sum_latency_value = group['latency'].sum()
            sum_latency_shift_value = group['latency_shift_sum'].sum()
            latency_dict[name] = sum_latency_value
            latency_shift_dict[name] = sum_latency_shift_value
        list_of_latency_results.append(latency_dict)
        list_of_latency_shift_results.append(latency_shift_dict)
    merged_latency_dict = defaultdict(list)
    merged_latency_shift_dict = defaultdict(list)
    for data in list_of_latency_results:
        for key, value in data.items():
            merged_latency_dict[key].append(value)
    for data in list_of_latency_shift_results:
        for key, value in data.items():
            merged_latency_shift_dict[key].append(value)
    avg_latency_dict = {}
    avg_latency_shift_dict = {}
    for key, values in merged_latency_dict.items():
        avg_value = sum(values) / len(values)
        avg_latency_dict[f'M.USER.APP.{key}'] = avg_value
    for key, values in merged_latency_shift_dict.items():
        avg_value = sum(values) / len(values)
        avg_latency_shift_dict[f'M.USER.APP.{key}'] = avg_value
    current_plan_measured_latency_workflow = avg_latency_shift_dict
    current_average_latency_measured = 0
    current_average_latency_static = 0
    for key, values in avg_latency_shift_dict.items():
            current_average_latency_measured += values
    for key, values in avg_latency_dict.items():
        current_average_latency_static += values
    measured_delay_in_rate_experiment=current_average_latency_measured


    
    '''
    measured_delay_in_rate_experiment = 0
    with open(folder_results+'sim_trace_static_link.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row['message'].startswith('M'):
                measured_delay_in_rate_experiment += float(
                    row['latency']) + float(row['shiftime'])
    '''

    #print(new_plan, new_plan_fitness)
    # return new_plan, new_plan_fitness
    return new_plan, measured_delay_in_rate_experiment


def main_dynamic_lambda(untiltime, experimentId, degree, depth):

    threshold = 0
    shift_delay = {}
    # experimentId=4

    alloDe = '/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/examples/DynamicWorkload/data/appExperiment/' + \
        str(experimentId)+'apps_6modules_per_app/allocDefinition.json'
    appDe = '/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/examples/DynamicWorkload/data/appExperiment/' + \
        str(experimentId)+'apps_6modules_per_app/appDefinition.json'
    userDe = '/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/examples/DynamicWorkload/data/appExperiment/' + \
        str(experimentId)+'apps_6modules_per_app/usersDefinition.json'

    folder_results = Path("src/examples/DynamicWorkload/results")
    folder_results.mkdir(parents=True, exist_ok=True)
    folder_results = str(folder_results)+"/"

    # set up the network
    t = Topology()
    # In NX-lib there are a lot of Graphs generators
    t.G = nx.generators.balanced_tree(degree, depth)
    attPR_BW = {x: random.randint(1, 10) for x in t.G.edges()}
    nx.set_edge_attributes(t.G, name="PR", values=attPR_BW)
    nx.set_edge_attributes(t.G, name="BW", values=attPR_BW)
    attIPT = {x: 100 for x in t.G.nodes()}
    nx.set_node_attributes(t.G, name="IPT", values=attIPT)
    attRAM = {x: random.randint(200, 7000) for x in t.G.nodes()}
    # set cloud node size high
    attRAM[0] = 10000
    nx.set_node_attributes(t.G, name="RAM", values=attRAM)
    nodes = t.G.nodes()
    links = t.G.edges
    nodes_ram = attRAM

    '''
    nodes_ram=[]
    for i in range(0,len(nodes)):
        nodes_ram.append(t.get_nodes_att()[i]['RAM'])
    '''

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
    pop = JSONPopulation(
        name="Statical", json=dataPopulation, external_lambda=None)

    """
    SIMULATOR ENGINE
    """
    s = Sim(t, default_results_path=folder_results+"sim_trace")

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
    print("initial placement:", EP_placement)

    # transfer placement from list into JSONplacement for YAFS simulator
    for allocation in placementJson["initialAllocation"]:
        app = allocation["app"]
        module_num = int(allocation["module_name"].split("_")[1]) - 1
        allocation["id_resource"] = EP_placement[app][module_num]
    json_data = json.dumps(placementJson, indent=2)
    placementJson = json.loads(json_data)
    placement = JSONPlacement(name="Placement", json=placementJson)

    # create mapping between app's module and Topology's node
    module_node_mapping = {}
    for allocation in placementJson["initialAllocation"]:
        module_name = allocation["module_name"]
        id_resource = allocation["id_resource"]
        module_node_mapping[module_name] = id_resource

    gen_size = 50  # set generation size for GA.py
    population = 10  # set population size for GA.py
    iteration_continue = True
    flag = 0  # to detect whether the firt iteration
    iteration_time = 1
    updated_monitored_latency_edge = {}  # keep each edge's latesed measured latency
    current_plan_measured_latency_workflow = {}  # keep execurated plan's latency
    final_plan = EP_placement  # if there is no need to adapt, then final plan is the EP
    lambda_rate = [150, 100, 70, 30, 10]  # store the
  
    lambda_rate= [115, 137.49756634, 149.46827136, 145.31088913, 126.97070502, 103.02929498, 84.68911087, 80.53172864, 92.50243366, 115]
    final_plan_measured_latency_in_each_lr = []

    for lr in lambda_rate:
        print(' ')
        print("**************************lambda rate is**************************", lr)

        # transfer placement from list(plan which generated by GA or EP) into JSONplacement for YAFS simulator
        for allocation in placementJson["initialAllocation"]:
            app = allocation["app"]
            module_num = int(allocation["module_name"].split("_")[1]) - 1
            allocation["id_resource"] = EP_placement[app][module_num]
        json_data = json.dumps(placementJson, indent=2)
        placementJson = json.loads(json_data)
        placement = JSONPlacement(name="Placement", json=placementJson)

        iteration_continue = True
        iteration_time = 1

        # start iteration
        while iteration_continue and iteration_time < 5:
            print("---------Iteration Time "+str(iteration_time)+"---------")
            iteration_time += 1
            sim_parameters = {"topology": t, "population": pop, "selectorPath": selectorPath, "apps": apps, "folder_results": folder_results, "untiltime": untiltime, "old_placement_json":placementJson }

            # create the simuator
            s = Sim(t, default_results_path=folder_results+"sim_trace")

            # create the dict to store each app's module
            app_modules_dict = {}
            for item in vars(placement)['data']['initialAllocation']:
                app = item['app']
                id_resource = item['id_resource']
                if app not in app_modules_dict:
                    app_modules_dict[app] = []
                app_modules_dict[app].append(id_resource)

            # set up the simulator, use our static lambda
            for aName in apps.keys():
                pop_app = JSONPopulation(
                    name="Statical_%s" % aName, json={}, external_lambda=lr)
                data = []
                for element in pop.data["sources"]:
                    if element['app'] == aName:
                        element['lambda'] = lr
                        data.append(element)
                    pop_app.data["sources"] = data
                s.deploy_app2(apps[aName], placement, pop_app, selectorPath)

            # run the simulator
            s.run(until=untiltime)

            # update the updated_monitored_latency_edge, keep the data of this dict is updated
            df = pd.read_csv(folder_results+'sim_trace_link.csv')
            df['latency_shift'] = df['latency'] + df['shiftime']
            average_latency_shift = df.groupby(['src', 'dst'])[
                'latency_shift'].mean()
            avg_latency_shift_dict = average_latency_shift.to_dict()
            updated_monitored_latency_edge.update(avg_latency_shift_dict)

            # get the executed plan's estimated(static) latency and measured latency, and get the measured latency of each workflow(我需要加上sender到第一个node的时间)
            list_of_dfs = []
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
            list_of_latency_results = []
            list_of_latency_shift_results = []
            for small_df in list_of_dfs:
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
                latency_dict = {}
                latency_shift_dict = {}
                for name, group in grouped:
                    sum_latency_value = group['latency'].sum()
                    sum_latency_shift_value = group['latency_shift_sum'].sum()
                    latency_dict[name] = sum_latency_value
                    latency_shift_dict[name] = sum_latency_shift_value
                list_of_latency_results.append(latency_dict)
                list_of_latency_shift_results.append(latency_shift_dict)
            merged_latency_dict = defaultdict(list)
            merged_latency_shift_dict = defaultdict(list)
            for data in list_of_latency_results:
                for key, value in data.items():
                    merged_latency_dict[key].append(value)
            for data in list_of_latency_shift_results:
                for key, value in data.items():
                    merged_latency_shift_dict[key].append(value)
            avg_latency_dict = {}
            avg_latency_shift_dict = {}
            for key, values in merged_latency_dict.items():
                avg_value = sum(values) / len(values)
                avg_latency_dict[f'M.USER.APP.{key}'] = avg_value
            for key, values in merged_latency_shift_dict.items():
                avg_value = sum(values) / len(values)
                avg_latency_shift_dict[f'M.USER.APP.{key}'] = avg_value
            current_plan_measured_latency_workflow = avg_latency_shift_dict
            current_average_latency_measured = 0
            current_average_latency_static = 0
            for key, values in avg_latency_shift_dict.items():
                current_average_latency_measured += values
            for key, values in avg_latency_dict.items():
                current_average_latency_static += values

            # get the initial plan's estimated(static) latency and measured latency if this is the first iteration（可能需要减去sender到第一个node的时间）
            if flag == 0:
                old_plan_measured_latency = current_average_latency_measured
                old_plan_estimated_latency = current_average_latency_static
                print("initial plan average measured latency:",
                      old_plan_measured_latency)
                print("initial plan average estimated latency:",
                      old_plan_estimated_latency)

            print("current plan's average static latency:",
                  current_average_latency_static)
            print("current plan's average measured latency:",
                  current_average_latency_measured)
            print("current plan's measued latency in each workflow",
                  current_plan_measured_latency_workflow)

            # judge is adapatation required
            is_adaptation_required = compare_avg_delay(
                threshold, folder_results+"sim_trace", untiltime, shift_delay)
            is_adaptation_required=True
            print("whether adaptation required:", is_adaptation_required)

            # start creating new plan by GA
            if is_adaptation_required:
                print("Start creating new plan!")
                ga = GA()
                new_plan, new_plan_fitness, new_plan_static_latency, adaptation_time = ga.run(nodes, app_list, modules, population, links, nodes_ram, modules_ram, msg, nx, t.G, attPR_BW,
                                                                                              gen_size, shift_delay, EP_placement, sim_parameters, updated_monitored_latency_edge, current_plan_measured_latency_workflow, untiltime, t, False, sender_msg_size)
                print("new plan created by GA:", new_plan)
                '''
                #this is another method which can move the deployment in the network
                for module_index, module_name in enumerate(module_node_mapping):
                    app_index = int(module_name.split("_")[0])
                    id_resource = new_plan[app_index][module_index%6]
                    module_node_mapping[module_name] = id_resource
                move_strategy_instance = MoveStrategy(module_node_mapping, False)# False means no movement, True means move
                move_strategy_instance(s, selectorPath)
                '''
                # calculate whether is worthy to adapt
                if flag == 0:
                    is_adaptation_worthy = get_effective_avg_latency(
                        new_plan_static_latency, untiltime, adaptation_time, folder_results+"sim_trace_link.csv", EP_placement, t.G)
                else:
                    is_adaptation_worthy = get_effective_avg_latency(
                        new_plan_static_latency, untiltime, adaptation_time, folder_results+"sim_trace_link.csv", new_plan, t.G)
                flag += 1
                print("whether adaptation worthy:", is_adaptation_worthy)

                if is_adaptation_worthy:
                    print("Start deploy new plan!")
                    print("new plan is:", new_plan)
                    final_plan = new_plan
                    '''
                    #another way to change the deployment on the network
                    move_strategy_instance = MoveStrategy(module_node_mapping, True)
                    move_strategy_instance(s, selectorPath)
                    '''
                    # change the placement into new_plan which created by GA
                    for allocation in placementJson["initialAllocation"]:
                        app = allocation["app"]
                        module_num = int(
                            allocation["module_name"].split("_")[1]) - 1
                        allocation["id_resource"] = new_plan[app][module_num]
                    json_data = json.dumps(placementJson, indent=2)
                    placementJson = json.loads(json_data)
                    placement = JSONPlacement(
                        name="Placement", json=placementJson)

                else:
                    iteration_continue = False
                    final_plan_measured_latency_in_each_lr.append(
                        current_average_latency_measured)
            else:
                iteration_continue = False
                final_plan_measured_latency_in_each_lr.append(
                    current_average_latency_measured)

    return final_plan_measured_latency_in_each_lr


if __name__ == '__main__':

    """
    Experiment 3: Different rate of message generation
    Lambda:
        increase regularly: [300, 500, 700, 900, 1100]
        change in a sine wave pattern
    Aim:
        Compare the performance bewteen GA-based Dynamic scheduling and GA-based Static scheduling
    """


    """
    # Dynamic scheduling
    original_stdout = sys.stdout
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    all_experiment_lr_latency = []
    for repeat_time in range(1, 6):
        sys.stdout = original_stdout
        print("Start Experiment ", repeat_time)
        with open(current_directory+'/results/experiments/lambda_rate/dynamic/output_sin_'+str(repeat_time)+'.txt', 'w') as file:
            sys.stdout = file
            print("Start dynamic Scheduling of different rate of message generation ")
            each_lr_latency = main_dynamic_lambda(4000, 6, 3, 4)
            all_experiment_lr_latency.append(each_lr_latency)
            print("each lr results:", each_lr_latency)
    sys.stdout = original_stdout
    print("Experiments of 5 times' output:", all_experiment_lr_latency)
    print("Simulation done!")
    """

    




    # Static scheduling
    static_scheduling_plan = []
    static_scheduling_fitness = []
    origin_plan = []
    origin_fitness = []
    original_stdout = sys.stdout
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    #lambda_rate = [10, 70, 150, 300, 600]
    lambda_rate = [150, 100, 70, 30, 10]
    lambda_rate= [115, 137.49756634, 149.46827136, 145.31088913, 126.97070502, 103.02929498, 84.68911087, 80.53172864, 92.50243366, 115]



    with open(current_directory+'/results/experiments/lambda_rate/static/output_sin.txt', 'w') as file:
        sys.stdout = file
        for lr in lambda_rate:
            print(' ')
            print("Start Static Scheduling of different rate of message generation ", lr)
            pointer = 0
            static_fitness_each_experiment = 0
            while pointer < 5:
                print("Start repeat time :", pointer+1)
                static_plan, static_fitness = main_static_lambda(4000, 6, 3, 4, lr)
                #static_fitness = main_static_lambda(4000, 6, 3, 5, 2800)
                
                
                print("static plan:", static_plan)
                print("static fitness:", static_fitness)
                static_fitness_each_experiment += static_fitness
                static_scheduling_plan.append(static_plan)
                pointer += 1
            static_scheduling_fitness.append(round(static_fitness_each_experiment/5, 4))
        print("GA-based static scheduling fitness of each message generation rate")
        print(static_scheduling_fitness)
        print("Scheduling plan in each lr")
        print(static_scheduling_plan)
    
    sys.stdout = original_stdout
    print("Simulation done!")





    

