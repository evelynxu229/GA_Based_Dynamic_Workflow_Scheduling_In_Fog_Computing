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
from yafs.GA_static import *
from yafs.EdgePlacement import *
from yafs.charts import *
from yafs.scenarios import *
from yafs.application import create_applications_from_json
from yafs.topology import Topology

from yafs.placement import JSONPlacement
from yafs.path_routing import DeviceSpeedAwareRouting
from yafs.distribution import deterministic_distribution,deterministicDistributionStartPoint
from collections import defaultdict


alloDe='/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/examples/DynamicWorkload/data/appExperiment/4apps_6modules_per_app/allocDefinition.json'
appDe='/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/examples/DynamicWorkload/data/appExperiment/4apps_6modules_per_app/appDefinition.json'
userDe='/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/examples/DynamicWorkload/data/appExperiment/4apps_6modules_per_app/usersDefinition.json'



#自定义策略
class CustomStrategy():

    def __init__(self,pathResults):
        self.activations = 0
        self.pathResults = pathResults

    #部署模块
    def deploy_module(self,sim,service,node):
        app_name = int(service[0:service.index("_")])
        app = sim.apps[app_name]
        services = app.services
        
        idDES = sim.deploy_module(app_name, service, services[service], [node])
        # with this `identifier of the DES process` (idDES) you can control it

    #非部署模块
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
    t.G = nx.generators.balanced_tree(3,6) # In NX-lib there are a lot of Graphs generators

    # Definition of mandatory attributes of a Topology，必要属性
    ## Attr. on edges，edge的必要属性设置，PR是channel Propagation speed, BW是Channel Bandwidth
    # PR and BW are 1 unit
    attPR_BW = {x: random.randint(1,10) for x in t.G.edges()}
    print(attPR_BW)
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
    #这是app的定义json，根据json建立app，需要整理出app的内在逻辑
    dataApp = json.load(open(appDe))

    
    apps = create_applications_from_json(dataApp)


    '''
    ###############################################################################################
    ####################################### MY CODE ###############################################
    ###############################################################################################
    '''

    #fetch list of nodes,获取一个节点列表
    nodes = t.G.nodes()

    #fetch network properties，获取网络属性
    #links between nodes，节点之间的links
    links = t.G.edges

    #node storage space，节点的RAM存储space被确定
    nodes_ram = attRAM

    #fetch list of applications，获取一个app列表，将json建立的app放到app_list中
    app_list = []
    for i in apps:
        app_list.append(i)

    #fetch modules and their ram, messages bytes and their source and dst
    modules = []
    modules_ram = []
    msg = [] #each entry is a list [[msg bytes, source, destination]]
    for i in dataApp:
        #ADD: check if application has module or not first!
        #把JSON的module和message提取出来
        app_module = i["module"]
        app_msgs = i["message"]

        #输出这个app有几个module
        print("app",i["id"],"has",len(app_module),"modules")


        mini_m = []
        mini_module_ram = []
        mini_msg = []

        #把每个module遍历一遍
        for m in app_module:
            print("modules are",m["name"]) #输出名字
            print("modules ram is", m['RAM']) #输出RAM size
            mini_m.append(m["name"]) #每个module的name放进mini_m表
            mini_module_ram.append(m["RAM"]) #每个module的ram放到列表中

        #把每个message遍历一遍，将其属性放到mini_msg列表中
        for m in app_msgs:
            print("messages source",m["s"], "dst",m["d"],"trans rate",m["bytes"])
            mini_msg.append([m["s"],m["d"],m["bytes"]])

        modules.append(mini_m)
        modules_ram.append(mini_module_ram)
        msg.append((mini_msg))

    print("nodes are", nodes)
    #nodes are [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
    print("list of apps are", app_list)
    #list of apps are [0, 1, 2, 3, 4]
    print("list of modules", modules)
    #list of modules [['0_01', '0_02', '0_03', '0_04', '0_05', '0_06'], ['1_01', '1_02', '1_03', '1_04', '1_05', '1_06'], ['2_01', '2_02', '2_03', '2_04', '2_05', '2_06'], ['3_01', '3_02', '2_03', '2_04', '2_05', '2_06'], ['4_01', '4_02', '4_03', '4_04', '4_05', '4_06']]
    print("list of modules and their rams", modules_ram)
    #list of modules and their rams [[1000, 3000, 3000, 1000, 2000, 700], [250, 500, 100, 400, 250, 50], [1000, 200, 63, 1000, 2500, 520], [2700, 700, 525, 320, 100, 1000], [500, 900, 1000, 3500, 750, 2200]]
    print("node ram:", nodes_ram)
    print("msgs s, dst and bytes",msg)
    #msgs s, dst and bytes [[['None', '0_01', 20], ['0_01', '0_02', 30], ['0_02', '0_03', 30], ['0_03', '0_04', 30], ['0_04', '0_05', 100], ['0_05', '0_06', 80]], [['None', '1_01', 20], ['1_01', '1_02', 30], ['1_02', '1_03', 10], ['1_03', '1_04', 300], ['1_04', '1_05', 65], ['1_05', '1_06', 15]], [['None', '2_01', 30], ['2_01', '2_02', 300], ['2_02', '2_03', 15], ['2_03', '2_04', 40], ['2_04', '2_05', 100], ['2_05', '2_06', 90]], [['None', '3_01', 20], ['3_01', '3_02', 20], ['3_02', '3_03', 42], ['3_03', '3_04', 15], ['3_04', '3_05', 87], ['3_05', '3_06', 230]], [['None', '4_01', 20], ['4_01', '4_02', 30], ['4_02', '4_03', 100], ['4_03', '4_04', 30], ['4_04', '4_05', 23], ['4_05', '4_06', 120]]]

    #上面就是填充好了modules，modules_ram以及msg三个列表

    ###########################################################################
    ########################## Start Evaluation ###############################
    ###########################################################################
    
    chart = charts()

    case = 7
    # run EdgePlacement ？？EdgePlacement是什么，这个模块可以控制其他模块的分配
    start = time.time() #这是开始时间

    edge_placement = EdgePlacement() #创建EdgePlacement对象

    
    EP_best = edge_placement.run(nodes, app_list, modules, links, nodes_ram, modules_ram, msg, nx, t.G, attPR_BW)
    
    return 0
    end = time.time() #这是结束时间
    runtime = end - start
    print("edge runtime:", runtime)

    population = 2
    gen_size = 50

    start = time.time()

    genetic = GA() #这是我们用的GA算法

    '''
    GA Run function Parameter:
        nodes: nodes from topology which created already
        app_list: app from JSON file
        modules: extract from app_list
        population: is a number, means ?
        links: edges from topology which created already
        nodes_ram: each nodes has ram, use for?
        modules_ram: each modules has ram, modules are from app
        msg: service? from app
        nx: networkx???? a library???
        t.G: a Graph which generated by Topology()
        attPR_BW: attribute of nodes
        generation:
    '''
    GA_best = genetic.run(nodes, app_list, modules, population, links, nodes_ram, modules_ram, msg, nx, t.G,
                          attPR_BW,
                          gen_size)
    end = time.time()
    runtime = end - start
    print("GA runtime:", runtime)

    print("GA fitness", GA_best)
    print("Edge fitness", 1 / EP_best)

    if case == 1:
        #这是第一个实验，通过改变初始population，对GA算法进行performance的测试
        #-------------------- CASE 1: vary population size -----------------------
        print("\nCase 1: Vary Population")

        population = [10,50,75,100]
        generation = 50
        GA_population = []
        for p in population:
            #run GA
            start = time.time()
            genetic = GA()
            GA_best = genetic.run(nodes, app_list, modules, p, links, nodes_ram, modules_ram,msg,nx,t.G,attPR_BW,generation)
            GA_population.append(GA_best)

            end = time.time()
            runtime = end-start
            print("runtime:",runtime)
            '''
            print("\n##########################################################")
            print("##########################################################")
            print("##########################################################")
            '''
        xTitle = "Population"
        fileName = "Case_1_Population_10x50x75x100"
        title = 'GA Latency Using Different Population Size'
        chart.trend(population, GA_population, fileName, title, xTitle)

        #draw a bar chart to compare the two fitness values
        fileName = "Case_1_Population_Best"
        title = 'GA vs Edge Placement Latency, Population Size Tuning'
        chart.bar(np.min(GA_population),1/EP_best,fileName, title)

    if case == 2:
        #改变不同的generation，测试GA的performance
        # -------------------- CASE 2: vary generation size -----------------------
        print("Case 2: Vary Generations")

        population = 50
        generation = [10,50,75,100]
        GA_generation= []

        '''
        for g in generation:
            # run GA
            start = time.time()

            genetic = GA()
            GA_best = genetic.run(nodes, app_list, modules, population, links, nodes_ram, modules_ram, msg, nx, t.G, attPR_BW,
                                  g)
            GA_generation.append(GA_best)

            end = time.time()
            runtime = end - start
            print("runtime:", runtime)
        '''

        GA_generation = [33.2, 11.5, 11, 10.9]
        EP_best = 1/49
        xTitle = "Generation"
        fileName = "Case_2_Generation_10x50x75x100"
        title = 'GA Latency Using Different Number of Generations'
        chart.trend(generation, GA_generation, fileName, title,xTitle)


        # draw a bar chart to compare the two fitness values
        fileName = "Case_2_Generation_Best"
        title = 'GA vs Edge Placement Latency, Number of Generations Tuning'
        chart.bar(np.min(GA_generation), 1 / EP_best, fileName, title)

    if case == 3:
        #关于超参数优化，就是最佳的网络size，对比GA和edge算法
        # -------------------- CASE 3: vary network size -----------------------
        print("Case 3: Vary Network")

        '''
        #change network size from the top manully
        # run GA

        population = 50
        generation = 50

        start = time.time()

        genetic = GA()
        GA_best = genetic.run(nodes, app_list, modules, population, links, nodes_ram, modules_ram, msg, nx, t.G, attPR_BW,
                              generation)
        end = time.time()
        runtime = end - start
        print("GA runtime:", runtime)

        print("GA fitness", GA_best)
        print("Edge fitness",1/EP_best)
        '''
        # network size (2,5) GA 12.637 s: 11, EP 0 s: 49
        # network size (2,7) GA 16.118 s: 17.3, EP 0.010 s: 97.273
        # network size (2,9) GA 27.72 s: 22, EP 0.033 s: 108.132
        # network size (2,15) GA: 22.75, EP 1.542: 139.665

        networkSize = [63,255,1023,65535]
        GA_best_list = [11, 17.3, 22, 22.75]
        EP_best_list = [49, 97.273, 108.132, 139.665]
        GA_time = [12.637, 16.118, 27.72, 1921.868]
        EP_time = [0.001, 0.01, 0.033, 1.542]

        xTitle = "Network Size"
        fileName = "Case_3_NetworkRuntime_63x255x1023x65535"
        title = 'GA vs Edge Placement Computation Time Using Different Network Sizes'
        chart.twoTrend(networkSize, GA_time,EP_time, fileName, title, xTitle)

    if case == 4:

        # -------------------- CASE 4: vary number of applications -----------------------
        print("Case 4: Vary Number of Applications")

        '''
        #change network size from the top manully
        # run GA

        population = 50
        generation = 50

        start = time.time()

        genetic = GA()
        GA_best = genetic.run(nodes, app_list, modules, population, links, nodes_ram, modules_ram, msg, nx, t.G, attPR_BW,
                              generation)
        end = time.time()
        runtime = end - start
        print("GA runtime:", runtime)

        print("GA fitness", GA_best)
        print("Edge fitness",1/EP_best)
        '''
        # recorded manually using 3 different JSON files found in data/appExperiment
        # ------ network size 63
        # ----------- 5 applications 6 modules
        # ---------------------- GA: 11.286     runtime: 14.296
        # ---------------------- EP: 29.929     runtime: 0.003
        # ----------- 8 applications 6 modules
        # ---------------------- GA: 11.5       runtime: 23.64
        # ---------------------- EP: 67.706     runtime: 0.002
        # ----------- 12 applications 6 modules
        # ---------------------- GA: 36.889     runtime: 42.898
        # ---------------------- EP: 116.487    runtime: 0.012

        # ------ network size 255
        # ----------- 5 applications 6 modules
        # ---------------------- GA: 22.27      runtime: 29.762
        # ---------------------- EP: 257.49     runtime: 0.008
        # ----------- 8 applications 6 modules
        # ---------------------- GA: 38.6       runtime: 38.608
        # ---------------------- EP: 257.49     runtime: 0.009
        # ----------- 12 applications 6 modules
        # ---------------------- GA: 54.786     runtime: 50.608
        # ---------------------- EP: 396.14     runtime: 0.017

        # ------ network size 1023
        # ----------- 5 applications 6 modules
        # ---------------------- GA: 13         runtime: 39.588
        # ---------------------- EP: 78.59      runtime:  0.013
        # ----------- 8 applications 6 modules
        # ---------------------- GA: 12.333        runtime: 47.649
        # ---------------------- EP: 136.466       runtime: 0.051
        # ----------- 12 applications 6 modules
        # ---------------------- GA: 36                 runtime: 72.442
        # ---------------------- EP: 239.466            runtime: 0.085

        numberOfApps = [5, 8, 12]
        GA_best_list_64 = [11.286, 11.5, 36.889]
        EP_best_list_64 = [29.929, 67.706, 116.487]

        GA_best_list_255 = [22.27, 38.6, 54.786]
        EP_best_list_255 = [257.49, 257.49, 396.14]

        GA_best_list_1023 = [13, 13, 15]
        EP_best_list_1023 = [29.929, 67.706, 239.466]

        xTitle = "Number of Applications"
        fileName = "Case_4_Number_of_Apps_5x8x12x63"
        title = 'GA vs Edge Placement Latency \n(Using Different Number of Applications, Network Size = 63 )'
        chart.twoBars(numberOfApps, GA_best_list_64, EP_best_list_64, fileName, title, xTitle)

        xTitle = "Number of Applications"
        fileName = "Case_4_Number_of_Apps_5x8x12x1023"
        title = 'GA vs Edge Placement Latency \n(Using Different Number of Applications, Network Size = 1023 )'
        chart.twoBars(numberOfApps, GA_best_list_1023, EP_best_list_1023, fileName, title, xTitle)

    if case == 5:
        # -------------------- CASE 5: vary number of modules -----------------------
        print("Case 5: Vary Number of Modules")

        '''
        # change network size from the top manully
        # run GA

        population = 50
        generation = 50

        start = time.time()

        genetic = GA()
        GA_best = genetic.run(nodes, app_list, modules, population, links, nodes_ram, modules_ram, msg, nx, t.G,
                              attPR_BW,
                              generation)
        end = time.time()
        runtime = end - start
        print("GA runtime:", runtime)

        print("GA fitness", GA_best)
        print("Edge fitness", 1 / EP_best)
        '''
        # recorded manually using 3 different JSON files found in data/appExperiment
        # ------ network size 63
        # ----------- 8 applications 6 modules
        # ---------------------- GA: 11      runtime: 30.753
        # ---------------------- EP: 67.706       runtime: 0.004
        # ----------- 8 applications 10 modules
        # ---------------------- GA: 41.042        runtime: 58.755
        # ---------------------- EP: 126.566     runtime: 0.006
        # ----------- 8 applications 15 modules
        # ---------------------- GA: 180.66       runtime: 134.387
        # ---------------------- EP: 396.259       runtime: 0.009


        numberOfApps = [6, 10, 15]
        GA_best_list_64 = [11, 41.042, 180.66]
        EP_best_list_64 = [67.706, 126.566, 396.259]


        xTitle = "Number of Applications"
        fileName = "Case_5_Number_of_Modules_6x10x15x63"
        title = 'GA vs Edge Placement Latency \n(Using Different Number of Modules, Network Size = 63 )'
        chart.twoBars(numberOfApps, GA_best_list_64, EP_best_list_64, fileName, title, xTitle)

    if case == 6:
        # -------------------- CASE 6: vary number of applications and modules -----------------------
        print("Case 6: Vary Number of Applications and Modules")


        '''
        # change network size from the top manully
        # run GA

        population = 50
        generation = 50

        start = time.time()

        genetic = GA()
        GA_best = genetic.run(nodes, app_list, modules, population, links, nodes_ram, modules_ram, msg, nx, t.G,
                              attPR_BW,
                              generation)
        end = time.time()
        runtime = end - start
        print("GA runtime:", runtime)

        print("GA fitness", GA_best)
        print("Edge fitness", 1 / EP_best)
        '''

        # recorded manually using 3 different JSON files found in data/appExperiment
        # ------ network size 63
        # ----------- 5 applications 6 modules
        # ---------------------- GA: 11.286         runtime: 18.78
        # ---------------------- EP: 29.929         runtime: 0.001
        # ----------- 5 applications 10 modules
        # ---------------------- GA: 18.765         runtime: 29.219
        # ---------------------- EP: 50.429         runtime: 0.003
        # ----------- 5 applications 15 modules
        # ---------------------- GA: 87.807         runtime: 65.293
        # ---------------------- EP: 260.224        runtime: 0.008

        # ----------- 8 applications 6 modules
        # ---------------------- GA: 11             runtime: 30.753
        # ---------------------- EP: 67.706         runtime: 0.004
        # ----------- 8 applications 10 modules
        # ---------------------- GA: 41.042         runtime: 58.755
        # ---------------------- EP: 126.566        runtime: 0.006
        # ----------- 8 applications 15 modules
        # ---------------------- GA: 180.66         runtime: 134.387
        # ---------------------- EP: 396.259        runtime: 0.009

        # ----------- 12 applications 6 modules
        # ---------------------- GA: 36.889         runtime: 53.982
        # ---------------------- EP: 116.487        runtime: 0.001
        # ----------- 12 applications 10 modules
        # ---------------------- GA: 175.368        runtime: 130.565
        # ---------------------- EP: 308.575        runtime: 0.016
        # ----------- 12 applications 15 modules
        # ---------------------- GA: 201.848         runtime: 231.275
        # ---------------------- EP: 560.636         runtime: 0.027

        numberOfModules = [6, 10, 15]
        GA_app_5 = [11.286, 18.765, 87.807]
        GA_app_8 = [11, 41.042, 180.66]
        GA_app_12 = [36.889, 175.368, 201.848]

        xTitle = "Number of Applications"
        fileName = "Case_6_Number_of_Apps_and_Mods_GA_x63"
        title = 'GA Latency \n(Using Different Number of Applications and Modules, Network Size = 63)'
        chart.threeBars(numberOfModules, GA_app_5, GA_app_8,GA_app_12, fileName, title, xTitle)

        EPA_app_5 = [29.929, 50.429, 260.224]
        EPA_app_8 = [67.706, 126.566, 396.259]
        EPA_app_12 = [116.487, 308.575, 560.636]

        xTitle = "Number of Applications"
        fileName = "Case_6_Number_of_Apps_and_Mods_EPA_x63"
        title = 'EPA Latency \n(Using Different Number of Applications and Modules, Network Size = 63)'
        chart.threeBars(numberOfModules, EPA_app_5, EPA_app_8, EPA_app_12, fileName, title, xTitle)

        xTitle = "Number of Applications"
        fileName = "Case_6_Number_of_Apps_and_Mods_GAVSEPA_x63"
        title = 'GA vs EPA Latency \n(Using Different Number of Applications and Modules, Network Size = 63)'
        chart.twoRowBars(numberOfModules, GA_app_5, GA_app_8,GA_app_12, EPA_app_5, EPA_app_8, EPA_app_12, fileName, title, xTitle)

    if case == 7:
        # -------------------- CASE 7: hyperparameter optemization -----------------------
        print("Case 7: Hyperparameter Optemization")

        #run both algorithms on a 63-node network, moderately dense with 8 applications 10 modules
        #tune GA hyper parameters: Population size, number of generations
        #using simple grid search
        '''
        population = [10, 50, 75, 100]
        generation = [10, 50, 75, 100]
        GA_population = []
        for p in population:
            for g in generation:
                # run GA
                start = time.time()
                genetic = GA()
                GA_best = genetic.run(nodes, app_list, modules, p, links, nodes_ram, modules_ram, msg, nx, t.G, attPR_BW,
                                      g)
                GA_population.append(GA_best)

                end = time.time()
                runtime = end - start
                print("GA runtime:", runtime)
                print("GA fitness (population", p,"generation",g,") is:",GA_best)

        print("Edge fitness", 1 / EP_best)
        '''
        # ----------- 8 applications 10 modules
        # ---------------------- EP: 126.565         runtime: 0.006

        # ---------------------- Population = 10
        # ---------------------- GA Generation 10: 82.114              runtime: 0.722
        # ---------------------- GA Generation 50: 185.992             runtime: 3.341
        # ---------------------- GA Generation 75: 61.064              runtime: 4.923
        # ---------------------- GA Generation 100: 62.497             runtime: 5.499

        # ---------------------- Population = 50
        # ---------------------- GA Generation 10: 101.601              runtime: 11.881
        # ---------------------- GA Generation 50: 79.818              runtime: 55.671
        # ---------------------- GA Generation 75: 72.125            runtime: 85.902
        # ---------------------- GA Generation 100: 109.534            runtime: 147.617

        # ---------------------- Population = 75
        # ---------------------- GA Generation 10: 81.346              runtime: 36.389
        # ---------------------- GA Generation 50: 95.722             runtime: 146.927
        # ---------------------- GA Generation 75: 30.8333              runtime: 211.308
        # ---------------------- GA Generation 100: 95.172             runtime: 95.173

        # ---------------------- Population = 100
        # ---------------------- GA Generation 10: 55.355              runtime: 55.661
        # ---------------------- GA Generation 50: 82.779              runtime: 252.88
        # ---------------------- GA Generation 75: 101.214              runtime: 485.405
        # ---------------------- GA Generation 100: 76.252             runtime: 576.585


        #run the GA using the hyperparameters population
        population = 10
        generation = 10
        start = time.time()
        genetic = GA()
        GA_best = genetic.run(nodes, app_list, modules, population, links, nodes_ram, modules_ram, msg, nx, t.G, attPR_BW,
                              generation)

        end = time.time()
        runtime = end - start
        print("GA runtime:", runtime)
        print("GA fitness", GA_best)
        return

    if case == 8:
        # -------------------- CASE 8: vary number of applications and modules -----------------------
        print("Case 8: Final Results")
        # ---------------------- Suggested EP: 126.565         runtime: 0.006

        # ---------------------- Suggested GA: 31.1         runtime: 43.105

        Algorithms = ["GA", "EPA"]
        GA_Prediction = 31.1
        EP_Prediction = 126.565

        fileName = "Final_latency"
        title = 'GA vs Edge Placement Latency'
        chart.bar(GA_best, 1/EP_best, fileName, title)

    #adjust placement of modules according to GA results
    '''
    ###############################################################################################
    #################################### END OF MY CODE ###########################################
    ###############################################################################################
    '''

    """
    SERVICE PLACEMENT 
    """
    placementJson = json.load(open(alloDe))
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
        dist = deterministic_distribution(100, name="Deterministic")
        idDES = s.deploy_source(app_name, id_node=node, msg=msg, distribution=dist)


    """
    This internal monitor in the simulator (a DES process) changes the sim's behaviour. 
    You can have multiples monitors doing different or same tasks.
    
    In this case, it changes the service's allocation, the node where the service is.
    """
    dist = deterministicDistributionStartPoint(stop_time/4., stop_time/2.0/10.0, name="Deterministic")
    evol = CustomStrategy(folder_results)
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
