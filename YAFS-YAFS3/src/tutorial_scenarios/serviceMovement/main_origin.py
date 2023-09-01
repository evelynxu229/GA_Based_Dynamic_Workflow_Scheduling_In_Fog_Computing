"""
    In this simulation, the module/service of the apps changes the allocation on the nodes randomly each time.


    @author: Isaac Lera
"""
import os
import select
import time
import json
import random
import logging.config
import csv

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

import sys
sys.path.insert(0,'/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/')
print(sys.path)

from yafs.core import Sim
from yafs.application import create_applications_from_json
from yafs.topology import Topology

from yafs.placement import JSONPlacement
from yafs.path_routing import DeviceSpeedAwareRouting
from yafs.distribution import deterministic_distribution,deterministicDistributionStartPoint
from collections import defaultdict




class CustomStrategy():

    def __init__(self,pathResults):
        self.activations = 0
        self.pathResults = pathResults

    def deploy_module(self,sim,service,node):
        app_name = int(service[0:service.index("_")])
        app = sim.apps[app_name]
        services = app.services
        idDES = sim.deploy_module(app_name, service, services[service], [node])
        # with this `identifier of the DES process` (idDES) you can control it

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

        # Clearing other related structures
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
        logging.info("Activating Custom process - number %i " % self.activations)
        self.activations += 1
        routing.invalid_cache_value = True # when the service change the cache of the Path.routing is outdated.

        # Current deployed services
        # print("Current deployed services> module:list(nodes)")
        current_services = self.get_current_services(sim)
        # print(current_services)

        # We move all the service to other random node
        for service in current_services:
            for currentNode in current_services[service]:
                newNode = random.sample(sim.topology.G.nodes(),1)[0]
                if not self.is_already_deployed(sim,service,newNode):
                    self.undeploy_module(sim,service,currentNode)
                    self.deploy_module(sim,service,newNode)
                    logging.info("Moving Service %s from %s to %s node"%(service,currentNode,newNode))


class RecordStrategy():

    def __init__(self,pathResults,t):
        self.activations = 0
        self.pathResults = pathResults
        self.t=t
        self.csv_writer = csv.writer(open("%sfitness_record.csv" % self.pathResults, 'w'))
        self.csv_writer.writerow(["time", "node", "processed_messages", "queue_length"])
        self.csv_reader = csv.reader(open("%ssim_trace.csv" % self.pathResults, 'r'))
        

    def get_average_queue_delay(self, current_time):
        current_time=100
        time_start=(current_time/1000)*1000
        time_end=time_start+100
        print("time_start", time_start)
        print("time_end", time_end)
        arrive_sum=0
        service_sum=0
        node_list=[]
        next(self.csv_reader)

        for row in self.csv_reader:
            print(row[14])
            print(type(row[14]))
            time_recep=float(row[14])
            time_out=float(row[12])
            if time_recep>time_end:
                break
            if time_recep>=time_start and time_recep<=time_end:
                arrive_sum+=1
            if time_out>=time_start and time_out<=time_end:
                service_sum+=1
            node_list.append(row[3])

        print(node_list)
        node_sum=len(list(set(node_list)))
        print(node_sum)
        arrive_rate=arrive_sum/node_sum
        service_rate=service_sum/node_sum
        p=arrive_rate/service_rate
        queue_delay=(p*p/(1-p))/arrive_rate
        return queue_delay
    
    def get_average_propagration(self, current_time):
        time_start=(current_time/10)*10
        time_end=(current_time/10+1)-1
        node_list=[]
        propagation_sum=0
        next(self.csv_reader)
        for row in self.csv_reader:
            time_recep=int(row[14])
            if time_recep>=time_start and time_recep<=time_end:
                src=int(row[7])
                dst=int(row[8])
                link=(src,dst)
                propagation_sum=propagation_sum+self.t.get_edge(link)[Topology.LINK_PR]
                node_list.append(row[3])
        node_sum=len(list(set(node_list)))
        average_propagration=propagation_sum/node_sum
        return average_propagration


    def __call__(self, sim, routing):
        print("Activating Record process - number %i " % self.activations)
        self.activations += 1
        current_time = sim.env.now
        ave_queue_delay=self.get_average_queue_delay( current_time)
        static_ave_propagration=self.get_average_propagration( current_time)
        ave_propagration=ave_queue_delay+static_ave_propagration
        self.csv_writer.writerow([current_time, ave_queue_delay, static_ave_propagration, ave_propagration])
        self.csv_writer.flush()




def main(stop_time, it,folder_results):

    """
    TOPOLOGY
    """
    t = Topology()

    # You also can create a topology using JSONs files. Check out examples folder
    size = 5
    t.G = nx.generators.binomial_tree(size) # In NX-lib there are a lot of Graphs generators
    

    # Definition of mandatory attributes of a Topology
    ## Attr. on edges
    # PR and BW are 1 unit
    attPR_BW = {x: 1 for x in t.G.edges()}
    nx.set_edge_attributes(t.G, name="PR", values=attPR_BW)
    nx.set_edge_attributes(t.G, name="BW", values=attPR_BW)
    ## Attr. on nodes
    # IPT
    attIPT = {x: 100 for x in t.G.nodes()}
    nx.set_node_attributes(t.G, name="IPT", values=attIPT)

    nx.write_gexf(t.G,folder_results+"graph_binomial_tree_%i.gexf"%size) # you can export the Graph in multiples format to view in tools like Gephi, and so on.

    print(t.G.nodes()) # nodes id can be str or int
    


    """
    APPLICATION or SERVICES
    """
    dataApp = json.load(open('/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/examples/Tutorial_JSONModelling/case/appDefinition.json'))
    apps = create_applications_from_json(dataApp)

    """
    SERVICE PLACEMENT 
    """
    placementJson = json.load(open('/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/examples/Tutorial_JSONModelling/case/allocDefinition.json'))
    placement = JSONPlacement(name="Placement", json=placementJson)

    """
    Defining ROUTING algorithm to define how path messages in the topology among modules
    """
    selectorPath = DeviceSpeedAwareRouting()

    """
    SIMULATION ENGINE
    """
    s = Sim(t, default_results_path=folder_results+"sim_trace")

    """
    Deploy services == APP's modules
    """
    for aName in apps.keys():
        s.deploy_app(apps[aName], placement, selectorPath)

    """
    Deploy users
    """
    userJSON = json.load(open('/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/examples/Tutorial_JSONModelling/case/usersDefinition.json'))
    for user in userJSON["sources"]:
        app_name = user["app"]
        app = s.apps[app_name]
        msg = app.get_message(user["message"])
        node = user["id_resource"]
        dist = deterministic_distribution(100, name="Deterministic")
        idDES = s.deploy_source(app_name, id_node=node, msg=msg, distribution=dist)


    """
    This internal monitor in the simulator (a DES process) changes the sim's behaviour. 
    You can have multiples monitors doing different or same tasks.
    
    In this case, it changes the service's allocation, the node where the service is.
    """
    '''
    dist = deterministicDistributionStartPoint(stop_time/4., stop_time/2.0/10.0, name="Deterministic")
    evol = CustomStrategy(folder_results)
    s.deploy_monitor("RandomAllocation",
                     evol,
                     dist,
                     **{"sim": s, "routing": selectorPath}) # __call__ args
    '''

    #dist = deterministic_distribution(stop_time/4., stop_time/2.0/10.0, name="Deterministic")
    #dist = deterministic_distribution(stop_time/4., name="Deterministic")
    #record = RecordStrategy(folder_results,t)
    #s.deploy_monitor("RecordMonitor", record, dist, **{"sim": s, "routing": selectorPath})



    """
    RUNNING - last step
    """
    logging.info(" Performing simulation: %i " % it)
    s.run(stop_time)  # To test deployments put test_initial_deploy a TRUE
    s.print_debug_assignaments()
    


if __name__ == '__main__':

    LOGGING_CONFIG = Path(__file__).parent / 'logging.ini'
    logging.config.fileConfig(LOGGING_CONFIG)

    folder_results = Path("src/tutorial_scenarios/serviceMovement/results/")
    folder_results.mkdir(parents=True, exist_ok=True)
    folder_results = str(folder_results)+"/"

    nIterations = 1  # iteration for each experiment
    simulationDuration = 20000

    # Iteration for each experiment changing the seed of randoms
    for iteration in range(nIterations):
        random.seed(iteration)
        logging.info("Running experiment it: - %i" % iteration)

        start_time = time.time()
        main(stop_time=simulationDuration,
             it=iteration,folder_results=folder_results)

        print("\n--- %s seconds ---" % (time.time() - start_time))

    print("Simulation Done!")



    # Analysing the results. 
    dfl = pd.read_csv(folder_results+"sim_trace"+"_link.csv")
    print("Number of total messages between nodes: %i"%len(dfl))

    df = pd.read_csv(folder_results+"sim_trace.csv")
    print("Number of requests handled by deployed services: %i"%len(df))

    dfapp0 = df[df.app == 0].copy() # a new df with the requests handled by app 0
    print(dfapp0.head())

    print("Different nodes where the app0 is deployed")
    print(np.unique(dfapp0["TOPO.dst"]))
    
    print("Number of requests handled at each position: \nid_node - requests")
    print(dfapp0.groupby(["TOPO.dst"])["id"].count())