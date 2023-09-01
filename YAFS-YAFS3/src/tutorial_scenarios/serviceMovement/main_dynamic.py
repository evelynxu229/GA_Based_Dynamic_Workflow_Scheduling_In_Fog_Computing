"""
    In this simulation, the module/service of the apps changes the allocation on the nodes randomly each time.


    @author: Isaac Lera
"""
import os
import time
import json
import random
import logging.config


import networkx as nx
from pathlib import Path


import sys
sys.path.insert(0,'/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/')
print(sys.path)


from yafs.core import Sim
from yafs.GA import *
from yafs.EdgePlacement import *
from yafs.charts import *
from yafs.scenarios import *
from yafs.application import create_applications_from_json
from yafs.topology import Topology

from yafs.placement import JSONPlacement
from yafs.path_routing import DeviceSpeedAwareRouting
from yafs.distribution import deterministic_distribution,deterministicDistributionStartPoint
from collections import defaultdict


alloDe='/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/tutorial_scenarios/serviceMovement/data/allocDefinition.json'
appDe='/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/tutorial_scenarios/serviceMovement/data/appDefinition.json'
userDe='/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/tutorial_scenarios/serviceMovement/data/usersDefinition.json'



#自定义策略
class CustomStrategy():

    def __init__(self,pathResults):
        self.activations = 0
        self.pathResults = pathResults

    #将指定service部署到指定模块
    def deploy_module(self,sim,service,node):
        app_name = int(service[0:service.index("_")])
        app = sim.apps[app_name]
        services = app.services
        idDES = sim.deploy_module(app_name, service, services[service], [node])
        # with this `identifier of the DES process` (idDES) you can control it

    #将指定service接触部署模块
    def undeploy_module(self,sim,service,node):
        app_name = int(service[0:service.index("_")])
        des = sim.get_DES_from_Service_In_Node(node, app_name, service)
        sim.undeploy_module(app_name, service,des)


    #已经部署的模块
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

    #获取当前的service
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
                    print("the new node is", newNode, "check if its already deployed..",
                          self.is_already_deployed(sim, service, newNode))
                    self.undeploy_module(sim,service,currentNode)
                    self.deploy_module(sim,service,newNode)
                    logging.info("Moving Service %s from %s to %s node"%(service,currentNode,newNode))


def main(stop_time, it):

    folder_results = Path("results/")
    folder_results.mkdir(parents=True, exist_ok=True)
    folder_results = str(folder_results)+"/"

    """
    TOPOLOGY，拓扑图
    """
    #创建一个拓扑图对象
    t = Topology()

    # You also can create a topology using JSONs files. Check out examples folder
    size = 32
    t.G = nx.generators.balanced_tree(2,5) # In NX-lib there are a lot of Graphs generators

    # Definition of mandatory attributes of a Topology，必要属性
    ## Attr. on edges，edge的必要属性设置，PR是channel Propagation speed, BW是Channel Bandwidth
    # PR and BW are 1 unit
    attPR_BW = {x: random.randint(1,10) for x in t.G.edges()}
    nx.set_edge_attributes(t.G, name="PR", values=attPR_BW)
    nx.set_edge_attributes(t.G, name="BW", values=attPR_BW)
    ## Attr. on nodes，nodes的必要属性设置
    # IPT
    attIPT = {x: 100 for x in t.G.nodes()}
    nx.set_node_attributes(t.G, name="IPT", values=attIPT)
    #RAM
    attRAM = {x: random.randint(200,10000) for x in t.G.nodes()}
    # set cloud node size high
    attRAM[0] = 500000
    nx.set_node_attributes(t.G, name="RAM", values=attRAM)


    nx.write_gexf(t.G,folder_results+"graph_binomial_tree_%i"%size) # you can export the Graph in multiples format to view in tools like Gephi, and so on.


    #fetch node info


    """
    APPLICATION or SERVICES
    """
    #这是app的定义json，根据json建立app
    dataApp = json.load(open(appDe))
    apps = create_applications_from_json(dataApp)


    """
    SERVICE PLACEMENT 
    """
    placementJson = json.load(open(alloDe))
    placement = JSONPlacement(name="Placement", json=placementJson)
    print("#########################################################")
    print(placement)
    print("#########################################################")

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
        print("deploying application number",aName,"name",apps[aName])
        s.deploy_app(apps[aName], placement, selectorPath)


    """
    Deploy users
    """
    userJSON = json.load(open(userDe))
    for user in userJSON["sources"]:
        app_name = user["app"]
        app = s.apps[app_name]
        msg = app.get_message(user["message"])
        print("p app",app,"p ms",msg)
        node = user["id_resource"]
        #dist只是用来模拟时间的，负责用来暂停多少时间
        dist = deterministic_distribution(100, name="Deterministic")
        #然后部署生产源就好了，这是根据JSON的静态部署，动态部署在DynamicWorkload中可以看到
        idDES = s.deploy_source(app_name, id_node=node, msg=msg, distribution=dist)


    """
    This internal monitor in the simulator (a DES process) changes the sim's behaviour. 
    You can have multiples monitors doing different or same tasks.
    
    In this case, it changes the service's allocation, the node where the service is.
    """
    dist = deterministicDistributionStartPoint(stop_time/4., stop_time/2.0/10.0, name="Deterministic")
    evol = CustomStrategy(folder_results)
    #想知道monitor什么时候可以被触发
    s.deploy_monitor("RandomAllocation",
                     evol,
                     dist,
                     **{"sim": s, "routing": selectorPath}) # __call__ args



    """
    RUNNING - last step
    """
    logging.info(" Performing simulation: %i " % it)
    s.run(stop_time)  # To test deployments put test_initial_deploy a TRUE
    s.print_debug_assignaments()


if __name__ == '__main__':

    #log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.ini')
    #logging.config.fileConfig(log_file_path)

    #print(os.getcwd())
    #logging.config.fileConfig(os.getcwd() + '/logging.ini')

    nIterations = 1  # iteration for each experiment
    simulationDuration = 20000

    # Iteration for each experiment changing the seed of randoms
    for iteration in range(nIterations):
        random.seed(iteration)
        logging.info("Running experiment it: - %i" % iteration)

        start_time = time.time()
        main(stop_time=simulationDuration,
             it=iteration)

        print("\n--- %s seconds ---" % (time.time() - start_time))

    print("Simulation Done!")
