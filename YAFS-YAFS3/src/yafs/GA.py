import random
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import sys
sys.path.insert(0,'/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/yafs/')
#print(sys.path)

from AdaptationCost import *
from temp_sim import *


from networkx.algorithms import approximation as approx

class GA:


    """
    Function:
        input is mainly a set of a module, and then return these module can be put in which nodes. return a list, it's random
    Args:
        self: GA itself)
        app: app id, e.g. 0
        nodes: Topology graph nodes, list of node id
        modules: list of modules id of one app
        links: no use
        nodes_ram: Topology graph each nodes ram
        modules_ram: app's modules's ram, e.g. app_modules_ram [1000, 3000, 3000, 1000, 2000, 700]
        nx: networkx
        G: Graph
    Returns:
        e.g.:
        application 4 modules ['4_01', '4_02', '4_03', '4_04', '4_05', '4_06'] are assigned to nodes [60, 29, 29, 6, 6, 2]
        old ram {0: 500000, 1: 1986, 2: 5138, 3: 9231, 4: 4969, 5: 2244, 6: 6149, 7: 4951, 8: 9052, 9: 3529, 10: 9165, 11: 7182, 12: 4912, 13: 7490, 14: 1701, 15: 9969, 16: 506, 17: 2694, 18: 9631, 19: 4166, 20: 3406, 21: 3212, 22: 3302, 23: 559, 24: 740, 25: 4460, 26: 8007, 27: 1331, 28: 1671, 29: 2333, 30: 2650, 31: 833, 32: 1514, 33: 207, 34: 2610, 35: 794, 36: 4715, 37: 7199, 38: 4058, 39: 3725, 40: 4580, 41: 7071, 42: 9697, 43: 4709, 44: 7582, 45: 2926, 46: 6055, 47: 1549, 48: 5513, 49: 2089, 50: 8169, 51: 9818, 52: 5693, 53: 3319, 54: 4181, 55: 465, 56: 4640, 57: 2119, 58: 3812, 59: 6295, 60: 730, 61: 5648, 62: 7181}
        current ram {0: 500000, 1: 1986, 2: 2938, 3: 9231, 4: 4969, 5: 2244, 6: 1899, 7: 4951, 8: 9052, 9: 3529, 10: 9165, 11: 7182, 12: 4912, 13: 7490, 14: 1701, 15: 9969, 16: 506, 17: 2694, 18: 9631, 19: 4166, 20: 3406, 21: 3212, 22: 3302, 23: 559, 24: 740, 25: 4460, 26: 8007, 27: 1331, 28: 1671, 29: 433, 30: 2650, 31: 833, 32: 1514, 33: 207, 34: 2610, 35: 794, 36: 4715, 37: 7199, 38: 4058, 39: 3725, 40: 4580, 41: 7071, 42: 9697, 43: 4709, 44: 7582, 45: 2926, 46: 6055, 47: 1549, 48: 5513, 49: 2089, 50: 8169, 51: 9818, 52: 5693, 53: 3319, 54: 4181, 55: 465, 56: 4640, 57: 2119, 58: 3812, 59: 6295, 60: 230, 61: 5648, 62: 7181}
    """
    def assign(self, nodes, app, modules, links, nodes_ram, modules_ram,nx,G):
        #print("in assign nodes are",nodes
        assignment = []
        currentRAM = copy.deepcopy(nodes_ram)
        #print("old ram is", nodes_ram)
        #print("current ram is", currentRAM)

        previous_assign = None #这个node节点是否被assign过

        #print("assigning modules", modules,"in application", app)

        #print("picking random node from 0 to", len(nodes))
        n = random.randrange(0,len(nodes))


        thresh = 15 #change later
        iteration = 0
        m = 0

        while m<len(modules):
            #获得这个node的当前剩余的ram
            n_ram = currentRAM.get(n)
            #获得这个modules的ram
            m_ram = modules_ram[m]

            #print("selected node is", n, "node ram is",n_ram)
            #print("module is",modules[m],"module ram is",m_ram)

            #检查该模块是否可以分配给节点的存储？
            #check if the module can be assigned to the node in terms of storage?
            #检查模块是否被适当地分配到节点上，考虑到节点和模块在网络上的位置
            #check if the module is assigned to a node appropriately in regards of
            #nodes and module position on the network

            #if this is the first time assigning just check the property of the node
            #QUESTION: why add a threshold condition? if we have to assign all modules
            #adding a threshold condition will leave some modules un-assigned
            if previous_assign is not None:
                #print("list of previously assigned are",assignment)
                #if previous node is placed in cloud, place the new node there as well if possible
                if previous_assign == 0:
                    #print("new node",previous_assign,"is a cloud node")
                    if n_ram >= m_ram:
                        #print("node ram is bigger than module ram")
                        # module can fit into this node
                        n = previous_assign
                        assignment.append(n)
                        currentRAM[n] = n_ram - m_ram
                        previous_assign = n
                        #n = random.randrange(0, len(nodes))
                        iteration += 1
                        m += 1
                        #print("assigned module to node")
                    else:
                        # start all over
                        #print("node ram is smaller than module ram")
                        #print("reset m counter")
                        n = random.randrange(0, len(nodes))
                        assignment = []
                        currentRAM = copy.deepcopy(nodes_ram)
                        m = 0
                        iteration = 0
                        previous_assign = None
                else:
                    #if there is previous assignment,
                    #select the node above it and check if it has enough space, if not go higher
                    #until you reach cloud, cloud RAM must be huge but how huge?
                    #print("last assignment is",assignment[-1])
                    for path in nx.all_simple_paths(G, source=assignment[-1], target=0):
                        #print("simple path", path)
                        for node_index in path:
                            flag_assign = False

                            #print("still going",node_index)
                            if node_index == previous_assign:
                                #if you come across the last assign node in the shortest path
                                indx = path.index(node_index)

                                n = node_index
                                n_ram = currentRAM.get(n)

                                #print("new node is", node_index, "its ram is",n_ram)

                                # try to place module on the same node if possible,
                                if n_ram >= m_ram:
                                    #print("node ram is bigger than module ram")
                                    # module can fit into this node
                                    assignment.append(n)
                                    previous_assign = n
                                    currentRAM[n] = n_ram-m_ram
                                    flag_assign = True
                                    #n = random.randrange(0, len(nodes))
                                    iteration += 1
                                    m += 1
                                    #print("assigned module to node")
                                else:
                                    #if not, fetch the next node in the shortest path
                                    #check if its not END OF LIST FIRST
                                    indx +=1

                                    while indx < len(path):
                                        next_node = path[indx]
                                        n = next_node
                                        n_ram = currentRAM.get(n)
                                        #print("next nodes is", next_node,"its ram is",n_ram)

                                        #try to place module on the next node if possible,
                                        if n_ram >= m_ram:
                                            #print("node ram is bigger than module ram")
                                            # module can fit into this node
                                            assignment.append(n)
                                            previous_assign = n
                                            currentRAM[n] = n_ram - m_ram
                                            flag_assign = True
                                            #n = random.randrange(0, len(nodes))
                                            iteration += 1
                                            m += 1
                                            #print("assigned module to node")

                                            break


                                        # if we cant place in cloud, start all over with a new random node
                                        if indx == len(path)-1:
                                            #print("you reached the cloud and couldnt assign the node")
                                            #print("reset m counter")
                                            n = random.randrange(0, len(nodes))
                                            assignment = []
                                            currentRAM = copy.deepcopy(nodes_ram)
                                            m = 0
                                            iteration = 0
                                            previous_assign = None


                                        #last condition, if we go through all upper nodes
                                        #and we couldn't assign, start all over
                                        indx +=1

                                if flag_assign == True:
                                    break


            else:
                #first time assigning, check ram condition if possible place
                if n_ram>=m_ram:
                    #print("node ram is bigger than module ram")
                    #module can fit into this node
                    assignment.append(n)

                    #update current ram for this node
                    currentRAM[n] = n_ram - m_ram
                    previous_assign = n
                    #n = random.randrange(0, len(nodes))
                    iteration += 1
                    m += 1
                    #print("assigned module to node")
                #if not possible go a node higher
                else:
                    # start all over
                    #print("node ram is smaller than module ram")
                    #print("reset m counter and RAM")
                    n = random.randrange(0, len(nodes))
                    assignment = []
                    currentRAM = copy.deepcopy(nodes_ram)
                    m = 0
                    iteration = 0
                    previous_assign = None

            '''
            if previous_assign is None:
                print("nram",n_ram,"mram",m_ram)
                if n_ram>=m_ram:
                    print("node ram is bigger than module ram")
                    #module can fit into this node
                    assignment.append(n)
                    previous_assign = n
                    n = random.randrange(0, len(nodes))
                    iteration += 1
                    m += 1
                    print("assigned module to node")

                else:
                    #start all over
                    print("node ram is smaller than module ram")
                    print("reset m counter")
                    n = random.randrange(0, len(nodes))
                    assignment = []
                    m = 0
                    iteration = 0
                    previous_assign = None

            #if there was a previous assignment check position property then check node property
            else:
                print("there is previous assignment")
                print("nram", n_ram, "mram", m_ram)
            '''


            #print("m counter is",m)

        #print("application", app, "modules", modules, "are assigned to nodes", assignment)

        #print("old ram",nodes_ram)
        #print("current ram", currentRAM)


        return assignment, currentRAM

    #section3.1.1 initial population, 创建representation of初代群体，其实就是每一个app的module在最初分配给了哪些fog nodes
    #需要figure out how to initialization, initialization means what?
    """
    Function:
        random initialize first generation, assign each module to some random nodes in the Topology
    Args:
        self: GA itself)
        nodes: Topology graph nodes
        apps: e.g.: app id [0, 1, 2, 3, 4]
        pop_size: population size ?
        links: Topology graph edges, with PR and BW, be setted in main.py through nx
        nodes_ram: each ram space in each nodes, be setted in main.py through nx
        modules_ram: be assigned in json file, five apps with 6 modules pre app
            e.g.: [[1000, 3000, 3000, 1000, 2000, 700], [250, 500, 100, 400, 250, 50], [1000, 200, 63, 1000, 2500, 520], [2700, 700, 525, 320, 100, 1000], [500, 900, 1000, 3500, 750, 2200]]
        nx: a library?
        G: Topology graph
    Returns:
        dimision is (pop_size,app_size,model_size), return 'pop_size' initial assigment situation.
    """
    def initialization(self, nodes, apps, modules, pop_size, links, nodes_ram, modules_ram,nx,G):

        population = []

        #进行pop_size次迭代,也就是为这些module分配50次，但是每一次都会更新nodes ram，也就是50个解决方案，并不是每一次都会叠加减少ram
        for i in range(pop_size):
            #print("\n\ncreating population number", (i+1))

            currentRAM = copy.deepcopy(nodes_ram)
            representation = []

            #便利每一个app
            for a in apps:
                #print("creating representations for application",a,"for population",(i+1))

                app_modules = []
                mini_modules_ram = []

                #fetch only the modules that belong to this application and their att

                #print("modules ram in assign", modules)

                modules_counter = 0
                for m in modules:

                    if m[0][1] == "_":
                        if m[0][0] == str(a):
                            app_modules = m
                            mini_modules_ram = modules_ram[modules_counter]
                        modules_counter += 1
                    else:
                        newM = m[0][0] + m[0][1]
                        if newM == str(a):
                            app_modules = m
                            mini_modules_ram = modules_ram[modules_counter]
                        modules_counter += 1
                
                #app_modules ['0_01', '0_02', '0_03', '0_04', '0_05', '0_06']
                #mini_modules_ram [1000, 3000, 3000, 1000, 2000, 700]

                #print("app_module",app_modules)
                #print("app_modules ram", mini_modules_ram)

                assignment, currentRAM = self.assign(nodes, a, app_modules, links, currentRAM, mini_modules_ram,nx,G)
                #print("assignments",assignment)
                representation.append(assignment)
            #print("representations are", representation)

            population.append(representation)
            #print("population is", population)
        #print("\n\ninitial population is")
        #print(population)
        return population
    

    def roulette(self,population, fitness_scores):
        generation = population.copy()
        fitness = fitness_scores.copy()


        # calculate fitness sum
        s = np.sum(fitness)

        # choose a random int between 0-sum
        r = random.uniform(0, s)

        # calculate incremental sum
        o_index = 0
        incremental_sum = 0
        #print("fitness scores", fitness)
        #print("fitness sum", s)
        #print("random number is", r)

        while o_index < len(fitness):
            incremental_sum += fitness[o_index]
            #print("incremental sum", incremental_sum)
            if incremental_sum >= r:
                #print("incremental sum is larger than r, stop")
                break
            o_index += 1

        #print("parent is", generation[o_index], "at index", o_index, "with fitness", fitness[o_index])

        parent = population[o_index]
        return parent

    """
    Function:
        fine the highest fitness score of part of the generations, and choose the father and mother 
    Args:
        self: GA itself
        population: the output from function initialization ==> 3D metrix dim: [generation num, app num, module num]
        fitness_scores: the fitness score of each generation ==> 2D list: [1, generation num]
    Returns:
        a generation with highest fitness score ==> list, called parent
    """
    def tournament(self, population, fitness_scores):
        #return the parent with the highest fitness score
        #print("fitness is ", fitness_scores)
        #print("population is", population)

        #select a random number of tournament pool
        t_size = random.randint(2,len(population))
        #print("tournament size",t_size)

        #choose t_size individuals from population to add to tournament
        #随机选一个pool，在这个pool中选择最大的fitness score
        t_pool = random.sample(range(0,len(population)),t_size)
        t_pool.sort()
        min_index = t_pool[0]
        min_indv = population[min_index]
        #print("fitness_scores:",fitness_scores)
        min_fitness = fitness_scores[min_index]


        #print("random sample is", t_pool)
        for i in t_pool:
            if fitness_scores[i] > min_fitness:
                min_index = i
                min_indv = population[i]
                min_fitness = fitness_scores[i]

            #print("at index", i)
            #print("idividual",population[i])
            #print("fitness", fitness_scores[i])

        #print("tournament winner", min_indv)
        #print("with fitness", min_fitness)
        #print("at index",min_index)

        parent = copy.deepcopy(min_indv)
        #print("parent:",parent)
        #parent: [[34, 3, 3, 3, 3, 1], [38, 38, 38, 38, 18, 18], [21, 21, 10, 10, 10, 10], [9, 4, 4, 1, 1, 0], [15, 7, 7, 0, 0, 0]]
        #choose the individual with the minimum latency

        return parent


    """
    aim: get each placement's fitness score
    placement:
    [
        app1: [53, 26, 26, 26, 12, 12, 12, 12, 12, 2], 
        app2: [62, 62, 62, 62, 62, 62, 62, 62, 62, 62], 
        app3: [2, 2, 2, 0, 0, 0, 0, 0, 0, 0], 
        app4: [61, 61, 61, 61, 61, 61, 30, 30, 30, 30], 
        app5: [62, 62, 62, 6, 6, 6, 6, 6, 6, 6], 
        app6: [11, 11, 11, 11, 11, 11, 11, 11, 11, 11], 
        app7: [45, 45, 22, 22, 22, 10, 10, 10, 10, 10], 
        app8: [30, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    parameter:
        old_plan_measured_latency_workflow: from execution, get from last execution
        updated_monitored_latency_edge: 每一次execution结束后，记录每一个edge的实际latency
        interval: simulator run time
        adatation_time: ？
        workflow_list:[] 需要知道每一个workflow的path是什么for path in nx.all_simple_paths(G, source=node_1, target=node_2):
        也就是一个app对应一个workflow。
        

    """
    def get_fitness_score(self, population, updated_monitored_latency_edge, old_plan_measured_latency_workflow, interval, nodes_ram, modules_ram, msg, G,BW,nx, shift_delay, old_placement, sim_parameters, t, sender_msg_size):
    
        generationSelect = copy.deepcopy(population)
        fitness_scores = []
        
        static_latency=[]
        adaptation_time=[]
        fitness = 0


        #print("population:", population)

        for gen in generationSelect:
            app_counter=0
            #print("该个体:", gen)
            workflow_path_list = []
            #calculate the adaptation cost of each generation
            calculator = AdaptationCostCalculator(old_placement, gen, G, BW, msg)
            adaptation_cost = calculator.calculate()
            adaptation_time.append(adaptation_cost)
    
            #get the path of each workflow, and corresponding bw and msg in this generation 
            #repBW_big = []
            #[11,12,13,14,15]
            
            for rep in gen:

                #这个rep的第一个头头，我需要计算我的sender

                workflow_path_list_app=[]
                counter = 0
                repBW = []
                repMSG = []
                rep_total_bw = 0
                if len(rep) > 1:
                    while counter < len(rep)-1:
                        node_1 = rep[counter]
                        node_2 = rep[counter+1]
                        if node_1 == node_2:
                            repBW.append(0)
                            workflow_path_list_app.append(None)
                        else:
                            for path in nx.all_simple_paths(G, source=node_1, target=node_2):
                                pc = 0
                                temp_list=[]
                                if len(path)>2:
                                    while pc < len(path)-1:
                                        link = (path[pc+1],path[pc])
                                        link_workflow=(path[pc],path[pc+1])
                                        temp_list.append(link_workflow)
                                        #workflow_path_list_app.append(link_workflow)
                                        pc+=1
                                        rep_total_bw = rep_total_bw + BW.get(link)
                                    workflow_path_list_app.append(temp_list)
                                else:
                                    while pc < len(path)-1:
                                        link = (path[pc+1],path[pc])
                                        link_workflow=(path[pc],path[pc+1])
                                        workflow_path_list_app.append(link_workflow)
                                        pc+=1
                                        rep_total_bw = rep_total_bw + BW.get(link)

                            repBW.append(rep_total_bw)
                        counter +=1
                else:
                    repBW.append(0)
                
                
                workflow_path_list.append(workflow_path_list_app)
                
            
                for i in msg:
                    counter = 0
                    miniMSG = []
                    if len(i) > 1:
                        while counter < len(i)-1:
                            miniMSG.append(i[counter+1][2])
                            counter += 1
                    else:
                        miniMSG.append(i[0][2])
                    repMSG.append(miniMSG)
                #repBW_big.append((repBW))
                app_counter+=1

            
            #print("该个体repMSG是：",repMSG)
            #repPR = copy.deepcopy(repBW_big)
            
            #according to the workflow path and it's bw and msg, calculate the static fitness






            #计算这个个体的，每一个app的sender到每一个app的第一个节点的产生的static latency
            sender_firstNode_static_delay_list=[]
            #sender_position=deployment_sender[app_counter]
            rep_counter=0
            for rep in gen:
                sender_firstNode_static_delay=0
                for path in nx.all_simple_paths(G, source=rep_counter+1, target=rep[0]):
                    pc = 0
                    while pc < len(path)-1:
                        link = (path[pc+1],path[pc])
                        bw=t.get_edge(link)['BW']
                        pr=t.get_edge(link)['PR']
                        #print("bw, pr:", bw, pr)
                        lat=sender_msg_size[rep_counter]*8 /(bw* 1000000.0) + pr
                        sender_firstNode_static_delay+=lat
                        pc+=1
                sender_firstNode_static_delay_list.append(sender_firstNode_static_delay)
                rep_counter+=1
            

            counter=0
            app_lat=[] #record each application's latency, is a 2d list
            static_fitness_dict={} # store each edge's static fitness
            for wf in workflow_path_list:
                #print("当前的wf是：", wf)
                edge_lat=[]
                inner_counter=0
                for edge in wf:
                    lat=0
                    if edge==None:
                        continue
                    msg_size=repMSG[counter][inner_counter]
                    #which means the source can't reach the destination directly, need trandfer
                    if isinstance(edge, list):
                        for ed in edge:
                            bw=t.get_edge(ed)['BW']
                            pr=t.get_edge(ed)['PR']
                            #print("bw, pr:", bw, pr)
                            lat=msg_size*8 /(bw* 1000000.0) + pr
                            
                            edge_lat.append(lat)
                            static_fitness_dict[ed]=lat
                    #don't need transfer
                    else:
                        bw=t.get_edge(edge)['BW']
                        pr=t.get_edge(edge)['PR']
                        lat=msg_size*8 /(bw* 1000000.0) + pr
                        
                        edge_lat.append(lat)
                        static_fitness_dict[edge]=lat

                    inner_counter+=1
                counter+=1
                
                app_lat.append(edge_lat)


            #这里让我们的app_lat增加上了我们的sender到第一个node的latency
            for i in range(0,len(app_lat)):
                app_lat[i].append(sender_firstNode_static_delay_list[i])

            #print("该个体的static latency是：", app_lat)
            
            #这是计算完这一个generation的latency了
            #predicted_static_latency is a 1d list, each item is the app's static latency 
            predicted_static_latency = []
            
           
            for item in app_lat:
                predicted_static_latency.append(sum(item))
                

            #print("predicted_static_latency:", predicted_static_latency)
            
            static_latency.append(sum(predicted_static_latency))
            #print("该群体的static_latency总和:", static_latency),这一部分我加上了我的sender到第一个node的latency

            static_fitness_workflow = app_lat
            #print("get each workflow's static fitness:", static_fitness_workflow)

            #print("get updated_monitored_latency_edge:", updated_monitored_latency_edge)

            predicted_delay_workflow=[]
            for app in workflow_path_list:
                #print("app:", app)
                predicted_delay_app=0
                for edge in app:
                    #print("edge:", edge)
                    if edge == None:
                        continue

                    elif isinstance(edge, list):
                        #print("进入list")
                        for ed in edge:
                            #print("这里执行了吗")
                            #print("ed:", ed)
                            if ed in updated_monitored_latency_edge:
                                #print("ed:", ed)
                                monitored_latency=updated_monitored_latency_edge[ed]
                                #print('monitored_latency:', monitored_latency)
                                predicted_latency=static_fitness_dict[ed]
                                #print('predicted_latency:', predicted_latency)
                                predicted_delay_edge=monitored_latency-predicted_latency
                                predicted_delay_app+=predicted_delay_edge
                            else:
                                predicted_delay_app+=0

                    else:
                        #print("这是个tuple")
                        if edge in updated_monitored_latency_edge:
                            #print("ed:", ed)
                            monitored_latency=updated_monitored_latency_edge[edge]
                            #print('monitored_latency:', monitored_latency)
                            predicted_latency=static_fitness_dict[edge]
                            #print('predicted_latency:', predicted_latency)
                            predicted_delay_edge=monitored_latency-predicted_latency
                            predicted_delay_app+=predicted_delay_edge
                        else:
                            predicted_delay_app+=0

                    #需要解决这个list, 应该在main函数中进行记录且更新
                    
                
                predicted_delay_workflow.append(predicted_delay_app)
            
            #这里需要将static_fitness_workflow转变成static_fitness_workflow_sum
            static_fitness_workflow_sum=[]
            for item in static_fitness_workflow:
                static_fitness_workflow_sum.append(sum(item))

            #print("static_fitness_workflow_sum:", static_fitness_workflow_sum)，这是一个计算上了从sender-第一个node的static latency的总latncy
            #print("predicted_delay_workflow:", predicted_delay_workflow)

            predicted_latency_workflow = [a + b for a, b in zip(static_fitness_workflow_sum, predicted_delay_workflow)]
    
            #predicted_latency_workflow=static_fitness_workflow_sum+predicted_delay_workflow
            #print("get each workflow's predicted fitness:",predicted_latency_workflow )

            #修改list的减法，让其减法可以成功
            #sum(a - b for a, b in zip(old_plan_measured_latency_workflow, predicted_latency_workflow))
            #print(old_plan_measured_latency_workflow)

            #这个old_plan_measured_latency传入的话需要更改，需要加上我们的第一个sender到第一个node的
            old_plan_measured_latency_workflow_value=list(old_plan_measured_latency_workflow.values())
            
            #print("旧方案workflow latency:", old_plan_measured_latency_workflow_value)
            #print("新方案workflow latency:", predicted_latency_workflow)
            #print(adaptation_cost)
            fitness=(sum(a - b for a, b in zip(old_plan_measured_latency_workflow_value, predicted_latency_workflow)))*interval - adaptation_cost
            
            #print("this plan‘s fitness socre:", fitness)
            #print("this plan's adatation cost is:", adaptation_cost)
            #print("this plan's interval is:", interval)
            fitness_scores.append(fitness)
            
            """
            这是一个作弊的做法，提供一个新的simulator，可供日后进行比较。
            #需要进行sim运行，然后更新shift_delay.
            temp_sim=new_simulator(sim_parameters["topology"], sim_parameters["population"], sim_parameters["selectorPath"], sim_parameters["apps"], sim_parameters["folder_results"],gen, sim_parameters["untiltime"],sim_parameters["old_placement_json"], shift_delay)
            temp_sim.simulator_run()

            shift_delay=temp_sim.update_shift_delay()
            
            estimated_delay, measured_delay=temp_sim.get_delay()
            measured_latency.append(measured_delay)
            estimated_latency.append(estimated_delay)

            calculator = AdaptationCostCalculator(old_placement, gen, G, bw, msg)
            adaptation_cost = calculator.calculate()
            print("adaptation cost:", adaptation_cost)
            fitness = 1/(measured_delay+adaptation_cost/80)
            fitness_scores.append(fitness)
            """

        '''
        print("GA(get_fitness_score)")
        print("fitness_scores",fitness_scores)
        print("measured_latency", measured_latency)
        print("estimated_latency", estimated_latency)
        '''
        #return fitness_scores, measured_latency, estimated_latency
        return fitness_scores, static_latency, adaptation_time
    

    def get_fitness_score_static(self, population,  msg, G,BW,nx, old_placement, t, sender_msg_size):
        
        generationSelect = copy.deepcopy(population)
        static_latency=[]
        adaptation_time=[]

        for gen in generationSelect:
            workflow_path_list = []
            #calculate the adaptation cost of each generation
            calculator = AdaptationCostCalculator(old_placement, gen, G, BW, msg)
            adaptation_cost = calculator.calculate()
            adaptation_time.append(adaptation_cost)
    
            #get the path of each workflow, and corresponding bw and msg in this generation 
            #repBW_big = []
            for rep in gen:
                workflow_path_list_app=[]
                counter = 0
                repBW = []
                repMSG = []
                rep_total_bw = 0
                if len(rep) > 1:
                    while counter < len(rep)-1:
                        node_1 = rep[counter]
                        node_2 = rep[counter+1]
                        if node_1 == node_2:
                            repBW.append(0)
                            workflow_path_list_app.append(None)
                        else:
                            for path in nx.all_simple_paths(G, source=node_1, target=node_2):
                                pc = 0
                                temp_list=[]
                                if len(path)>2:
                                    while pc < len(path)-1:
                                        link = (path[pc+1],path[pc])
                                        link_workflow=(path[pc],path[pc+1])
                                        temp_list.append(link_workflow)
                                        #workflow_path_list_app.append(link_workflow)
                                        pc+=1
                                        rep_total_bw = rep_total_bw + BW.get(link)
                                    workflow_path_list_app.append(temp_list)
                                else:
                                    while pc < len(path)-1:
                                        link = (path[pc+1],path[pc])
                                        link_workflow=(path[pc],path[pc+1])
                                        workflow_path_list_app.append(link_workflow)
                                        pc+=1
                                        rep_total_bw = rep_total_bw + BW.get(link)

                            repBW.append(rep_total_bw)
                        counter +=1
                else:
                    repBW.append(0)
                
                
                workflow_path_list.append(workflow_path_list_app)
                
            
                for i in msg:
                    counter = 0
                    miniMSG = []
                    if len(i) > 1:
                        while counter < len(i)-1:
                            miniMSG.append(i[counter+1][2])
                            counter += 1
                    else:
                        miniMSG.append(i[0][2])
                    repMSG.append(miniMSG)
                #repBW_big.append((repBW))



            #计算这个个体的，每一个app的sender到每一个app的第一个节点的产生的static latency
            sender_firstNode_static_delay_list=[]
            #sender_position=deployment_sender[app_counter]
            rep_counter=0
            for rep in gen:
                sender_firstNode_static_delay=0
                for path in nx.all_simple_paths(G, source=rep_counter+1, target=rep[0]):
                    pc = 0
                    while pc < len(path)-1:
                        link = (path[pc+1],path[pc])
                        bw=t.get_edge(link)['BW']
                        pr=t.get_edge(link)['PR']
                        #print("bw, pr:", bw, pr)
                        lat=sender_msg_size[rep_counter]*8 /(bw* 1000000.0) + pr
                        sender_firstNode_static_delay+=lat
                        pc+=1
                sender_firstNode_static_delay_list.append(sender_firstNode_static_delay)
                rep_counter+=1


            #print("该个体repMSG是：",repMSG)
            #repPR = copy.deepcopy(repBW_big)
            counter=0
            app_lat=[] #record each application's latency, is a 2d list
            static_fitness_dict={} # store each edge's static fitness
            #according to the workflow path and it's bw and msg, calculate the static fitness
            
            for wf in workflow_path_list:
                #print("当前的wf是：", wf)
                edge_lat=[]
                inner_counter=0
                for edge in wf:
                    lat=0
                    if edge==None:
                        continue
                    msg_size=repMSG[counter][inner_counter]
                    #which means the source can't reach the destination directly, need trandfer
                    if isinstance(edge, list):
                        for ed in edge:
                            bw=t.get_edge(ed)['BW']
                            pr=t.get_edge(ed)['PR']
                            #print("bw, pr:", bw, pr)
                            lat=msg_size*8 /(bw* 1000000.0) + pr
                            
                            edge_lat.append(lat)
                            static_fitness_dict[ed]=lat
                    #don't need transfer
                    else:
                        bw=t.get_edge(edge)['BW']
                        pr=t.get_edge(edge)['PR']
                        lat=msg_size*8 /(bw* 1000000.0) + pr
                        
                        edge_lat.append(lat)
                        static_fitness_dict[edge]=lat

                    inner_counter+=1
                counter+=1
                
                app_lat.append(edge_lat)
            
            #print("该个体的static latency是：", app_lat)

            for i in range(0,len(app_lat)):
                app_lat[i].append(sender_firstNode_static_delay_list[i])

            
            #这是计算完这一个generation的latency了
            #predicted_static_latency is a 1d list, each item is the app's static latency 
            predicted_static_latency = []
            
            for item in app_lat:
                predicted_static_latency.append(sum(item))

            static_latency.append(sum(predicted_static_latency))
            
        return static_latency







    """
    Function:
        calculate the fitness of each generation and choose next generation's parents
    Args:
        self: GA itself
        population: the output from function initialization
        nodes_ram: no use, cause is being updated
        modules_ram: no use, cause is being updated
        msg: consist by [source, destination, bytes]
        G: Topology
        bw: bandwith, is attPR_BW
        nx: networkx
        pop_flag: ?
    Intermediate param:
        param repMSG: record the MSG size of each App's path
        param repBW_big: record the bandwidth of each App's path
        param repPR: the copy one of repBW_big, but is the propagation??
        param fitness_store: store each generation's fitness score
    Returns:
        if pop_flag == False:
            father and mother ==》 two list, both generation
        else:
            ?
    """
    def select(self, population, nodes_ram, modules_ram, msg, G,bw,nx, shift_delay,old_placement,sim_parameters, updated_monitored_latency_edge,old_plan_measured_latency_workflow, interval,t, pop_flag , static_flag, sender_msg_size):
        
        #fitness_scores, latency_plan, estimated_latency=self.get_fitness_score(population, nodes_ram, modules_ram, msg, G,bw,nx, shift_delay,old_placement, sim_parameters)
        if static_flag==False:
            fitness_scores, all_gen_static_latency, adaptation_time=self.get_fitness_score(population,updated_monitored_latency_edge, old_plan_measured_latency_workflow,interval,  nodes_ram, modules_ram, msg, G,bw,nx, shift_delay,old_placement, sim_parameters, t, sender_msg_size)
        else:
            fitness_scores=self.get_fitness_score_static(population,  msg, G,bw,nx, old_placement, t, sender_msg_size)

        generationSelect = copy.deepcopy(population)

        #fitness calculation for generation
        if pop_flag == False:
            o_1 = self.tournament(population, fitness_scores)
            o_2 = self.tournament(population, fitness_scores)
        
            return o_1, o_2

        #fitness calculation for best fit
        else:
            if static_flag==False:
                val, idx = max((val, idx) for (idx, val) in enumerate(fitness_scores))
                return val, generationSelect[idx], all_gen_static_latency[idx], adaptation_time[idx]
            else:
                val, idx = min((val, idx) for (idx, val) in enumerate(fitness_scores))
                return val, generationSelect[idx]
    

    """
    Function:
        each crossover need to validate the ram space of the nodes
    Args:
        self: GA itself
        part_1: front k part of parent_1
        part_2: back k part of parent_2
        nodes: no use
        nodes_ram: each nodes' ram space
        modules: no use
        modules_ram: each modules' ram space
        parent: parent_1
    Returns:
        if can be validated, then use part_1+part_2 as child
        else use parent_1 + 100% mutation probability as child
       
    """
    def crossoverValidation(self,part_1,part_2, nodes, nodes_ram, modules, modules_ram,nx,G,parent):
        child = []
        mutation_prob = random.uniform(0, 1)
        valid = True

        #print("\nvalidation cross over")
        #print("part 1",part_1,"part 2",part_2)
        #print("nodes are",nodes)
        #print("complete nodes rams", nodes_ram)
        #print("modules",modules)
        #print("modules ram",modules_ram)

        #calculate current ram state after using up the first part
        currentRAM = copy.deepcopy(nodes_ram)

        counter = 0

        while counter < len(part_1):
            #print("part one elements",part_1[counter])
            inner_counter = 0
            while inner_counter < len(part_1[counter]):
                n = part_1[counter][inner_counter]
                n_ram = currentRAM.get(n)
                m_ram = modules_ram[counter][inner_counter]

                new_ram = n_ram - m_ram
                currentRAM[n] = new_ram

                #print("part one inner elements",n,"ram is",n_ram)
                #print("corrosponsing module is",modules[counter][inner_counter],"its ram is",m_ram)
                #print("new ram is", new_ram,"\n")
                inner_counter +=1

            counter +=1
            m = counter

        #print("current m counter", m)
        #print("current ram after part 1", currentRAM)

        #try assigning second part using the current ram state
        counter = 0
        valid = True
        while counter < len(part_2):
            #print("part two elements", part_2[counter])
            inner_counter = 0
            while inner_counter < len(part_2[counter]):
                n = part_2[counter][inner_counter]
                n_ram = currentRAM.get(n)
                m_ram = modules_ram[m][inner_counter]

                if n_ram >= m_ram:
                    new_ram = n_ram - m_ram
                    currentRAM[n] = new_ram

                    #print("part two inner elements", n, "ram is", n_ram)
                    #print("corrosponsing module is", modules[m][inner_counter], "its ram is", m_ram)
                    #print("new ram is", new_ram, "\n")

                else:
                    # if assignment can't be done?
                    valid = False
                    #print("couldnt assign module",modules[m][inner_counter],"to node",n)

                inner_counter +=1
            m += 1
            counter +=1
        #print("current ram after part 2", currentRAM)

        if valid == True:
            child = part_1 + part_2
            #print("crossover pass")
        else:
            #if crossover is not valid, make the child the first parent and increase mutation probability
            child = parent
            mutation_prob = 1
            #print("crossover fail")

        #print("new child is", child)

        return child, mutation_prob
    
    """
    Function:
        crossover happen in the parents generation
    Args:
        self: GA itself
        parent_1: output of function selection
        parent_2: output of function seleciton
        nodes: nodes id in the Topology
        nodes_ram: each nodes' ram space
        modules: modules id in each app
        modules_ram: each modules' ram space
    Returns:
        child and mutation probability
    """
    def crossover(self, parent_1, parent_2, nodes, nodes_ram, modules, modules_ram,nx,G):
        #one point crossover
        #random point
        k = 4
        #print("\n\nparent 1", parent_1)
        #print("parent 2", parent_2)

        part_1 = parent_1[0:k]
        part_2 = parent_2[k:]

        child, mutation_prob = self.crossoverValidation(part_1,part_2, nodes, nodes_ram, modules, modules_ram,nx,G,parent_1)

        if parent_1[k:] == parent_2[k:]:
            #print("second parts are identical")
            mutation_prob = 1

        return child, mutation_prob

    def mutate(self, child, mutation_prob,nodes_ram, modules_ram,G, nx):
        random_app = random.randrange(0,len(child))
        random_module = random.randrange(0,len(child[random_app]))
        mutation_thresh = 0.2
        currentRAM = copy.deepcopy(nodes_ram)


        #if mutation rate > threshold

        if mutation_prob > mutation_thresh:
            #if there is a mutation, randomly select either move 0 one node closer to cloud
            # or 1 one move closer to edge
            move = random.randint(0,1)

            #print("random movement is", move)
            #print("\nchild to mutate", child)
            #print("mutation rate", mutation_prob)

            #print("nodes ram is", nodes_ram)
            #print("modules ram is",modules_ram)
            counter = 0

            # get the ram and calculate the current ram state
            for a in child:
                inner_counter = 0
                #print("the application is",a)
                for n in a:
                    #print("the node is", n)
                    #print("node ram is", currentRAM.get(n))
                    #print("the module is",modules_ram[counter][inner_counter])
                    n_ram = currentRAM.get(n)
                    m_ram = modules_ram[counter][inner_counter]
                    currentRAM[n] = n_ram - m_ram
                    inner_counter +=1
                counter += 1

            # get the path from the selected node to cloud
            # try to move the node one node closer to cloud
            # check the next node and validate placement
            # check if ram allows it

            #print("random application", random_app, "is", child[random_app])
            #print("random module", random_module, "is", child[random_app][random_module])
            #print("corrosposing application modules ram is", modules_ram[random_app][random_module])

            random_node = child[random_app][random_module]
            #print("the random module is", random_module)
            #print("the random node is", random_node)

            if move == 0:
                #print("moving closer to cloud")
                '''
                for path in nx.all_simple_paths(G, source=random_node, target=0):
                    #print("next potential node",path[1])

                    n = path[1]

                    #if its last in list, just mutate
                    if random_module == len(child[random_app])-1:
                        #print("this is the last node in line")

                        #check if node ram is enough
                        n_ram = currentRAM.get(n)
                        m_ram = modules_ram[random_app][random_module]
                        if n_ram >= m_ram:
                            print("mutate node")
                            currentRAM[n] = n_ram - m_ram
                            child[random_app][random_module] = n

                        #if cant assign ram just return the same child

                    # if its not last in list check the node next to it to see if placement is possible
                    else:
                        adj = child[random_app][random_module+1]
                        #print("adj node assignment is", adj)
                        n_index = path.index(n)
                        adj_index = path.index(adj)
                        #print("potential index is", n_index,"adj index is", adj_index)
                        if n_index <= adj_index:
                            # check if node ram is enough
                            n_ram = currentRAM.get(n)
                            m_ram = modules_ram[random_app][random_module]
                            if n_ram >= m_ram:
                                print("mutate node")
                                currentRAM[n] = n_ram - m_ram
                                child[random_app][random_module] = n

                        #else if ram isnt enough, dont mutate


                    #else if placement isnt possible, dont mutate

                '''
                child = self.moveCloud(child, mutation_prob,nodes_ram, modules_ram,G, nx, random_node, random_module, random_app,currentRAM)
            else:
                #print("moving closer to edge")
                #print("edges are",G.edges(random_node),"length",len(G.edges(random_node)))
                #find if there is a node in a lower level
                #if there is no node at lower level this means its an edge node, try going upward?

                #once we get the node at lower level, check the placement condition, is the previous assignment contradicting
                #the new placement?

                #if its first in list, just mutate if ram is enough
                # if its last in list, just mutate to any previous node
                if random_module == 0:
                    #print("first module, just mutate")
                    # check if node ram is enough
                    '''
                    n_ram = currentRAM.get(n)
                    m_ram = modules_ram[random_app][random_module]
                    if n_ram >= m_ram:
                        print("mutate node")
                        currentRAM[n] = n_ram - m_ram
                        child[random_app][random_module] = n
                    '''
                    # if cant assign ram just return the same child

                #else its not first in list, check its previous node assignment
                    #if we placement is valid, check ram condition

            #print("mutated child is", child)
            #print("new ram is", currentRAM)


        return child

    def moveCloud(self, child, mutation_prob,nodes_ram, modules_ram,G, nx, random_node, random_module, random_app,currentRAM):
        #print("moving one node closer to cloud")
        for path in nx.all_simple_paths(G, source=random_node, target=0):
            # print("next potential node",path[1])

            n = path[1]

            # if its last in list, just mutate
            if random_module == len(child[random_app]) - 1:
                # print("this is the last node in line")

                # check if node ram is enough
                n_ram = currentRAM.get(n)
                m_ram = modules_ram[random_app][random_module]
                if n_ram >= m_ram:
                    #print("mutate node")
                    currentRAM[n] = n_ram - m_ram
                    child[random_app][random_module] = n

                # if cant assign ram just return the same child

            # if its not last in list check the node next to it to see if placement is possible
            else:
                adj = child[random_app][random_module + 1]
                # print("adj node assignment is", adj)
                n_index = path.index(n)
                adj_index = path.index(adj)
                # print("potential index is", n_index,"adj index is", adj_index)
                if n_index <= adj_index:
                    # check if node ram is enough
                    n_ram = currentRAM.get(n)
                    m_ram = modules_ram[random_app][random_module]
                    if n_ram >= m_ram:
                        #print("mutate node")
                        currentRAM[n] = n_ram - m_ram
                        child[random_app][random_module] = n

        return child

    def moveEdge(self, child, mutation_prob, nodes_ram, modules_ram, G, nx, random_node, random_module, random_app, currentRAM):
        #print("moving one node closer to edge")
        edges = G.edges(random_node)

        #print("child is", child)
        #print("random app is", random_app)
        #print("random module is", random_module)
        #print("random node is", random_node)
        #print("edge length is", len(edges))

        # if its first in list, just mutate
        if random_module == 0:
            #print("first module, just mutate")



            # if edge = 2, we are currently in cloud, choose either nodes
            if len(edges) == 2:
                small_counter = 0
                rand_choice = random.choice([0, 1])
                #print("our random choice is", rand_choice)
                for e in edges:
                    if small_counter == rand_choice:
                        n = e[1]
                    #print(e)
                    small_counter += 1

                #print("cc new node is", n)
                # check avilable storage

                n_ram = currentRAM.get(n)
                m_ram = modules_ram[random_app][random_module]
                if n_ram >= m_ram:
                    #print("mutate node")
                    currentRAM[n] = n_ram - m_ram
                    child[random_app][random_module] = n
                #print("alert")
                #i think code is complete here, check once you build a thorough scenario

            # if edge = 3, we are in the middle
            else:
                #print(edges)
                small_counter = 0
                #rand_choice = random.choice([1, 2])
                #print("our random choice isss", rand_choice)

                for e in edges:
                    if small_counter == 1:
                        n = e[1]
                    if small_counter == 2:
                        n_2 = e[1]

                    small_counter += 1

                #print("cc new node is", n)

                n_ram = currentRAM.get(n)
                m_ram = modules_ram[random_app][random_module]
                if n_ram >= m_ram:
                    #print("mutate node")
                    currentRAM[n] = n_ram - m_ram
                    child[random_app][random_module] = n
                else:
                    #print("cant place first node, trying node",n_2)
                    n_ram = currentRAM.get(n_2)
                    if n_ram >= m_ram:
                        #print("mutate node")
                        currentRAM[n] = n_ram - m_ram
                        child[random_app][random_module] = n
                    #else:
                        #print("storage condition violated")



        #check next node for placement condition
        else:
            previous_node = child[random_app][random_module-1]
            #print("previous node is",previous_node)

            if random_node == previous_node:
                child = child
                #print("the selected node is the same as the previous node, dont mutate")

            #previous node is differet, try mutating
            else:
                #get path from current node to previous node
                for path in nx.all_simple_paths(G, source=random_node, target=previous_node):
                    #print("path to going down is",path)

                    if len(path) == 2:
                        #print("the assignment should be on the previous node", path[1])
                        #try placement with storage condition
                        n = path[1]

                        n_ram = currentRAM.get(n)
                        m_ram = modules_ram[random_app][random_module]
                        if n_ram >= m_ram:
                            #print("mutate node")
                            currentRAM[n] = n_ram - m_ram
                            child[random_app][random_module] = n

                    # try placement with storage condition, if fails, try previous node and so on
                    else:
                        counter = 1
                        while counter < len(path):
                            #print("the assignment should be on the middle node", path[counter])
                            n = path[counter]
                            n_ram = currentRAM.get(n)
                            m_ram = modules_ram[random_app][random_module]
                            if n_ram >= m_ram:
                                #print("mutate node")
                                currentRAM[n] = n_ram - m_ram
                                child[random_app][random_module] = n

                                break

                            counter+=1

        return child
    """
    Function:
        about how to mutation
    Args:
        self: GA itself
        child: the output from crossover, and is a 
    Returns:
        
    """
    def newMutation(self, child, mutation_prob,nodes_ram, modules_ram,G, nx):
        random_app = random.randrange(0, len(child))
        random_module = random.randrange(0, len(child[random_app]))
        mutation_thresh = 0.8
        currentRAM = copy.deepcopy(nodes_ram)
        mutation = copy.deepcopy(child)

        # if mutation rate > threshold

        if mutation_prob > mutation_thresh:
            #print("Mutate child")
            # if there is a mutation, randomly select either move 0 one node closer to cloud
            # or 1 one move closer to edge

            child_validate = copy.deepcopy(mutation)
            counter = 0

            # get the ram and calculate the current ram state
            for a in mutation:
                inner_counter = 0
                for n in a:
                    n_ram = currentRAM.get(n)
                    m_ram = modules_ram[counter][inner_counter]
                    currentRAM[n] = n_ram - m_ram
                    inner_counter += 1
                counter += 1

            random_node = mutation[random_app][random_module]
            e_length = len(G.edges(random_node))


            #Condition 1:
            #if node is a cloud node, always go one node closer to edge
            if random_node == 0:
                #print("its a cloud node, move closer to edge")
                child = self.moveEdge(mutation, mutation_prob, nodes_ram, modules_ram, G, nx, random_node, random_module,
                                       random_app, currentRAM)


            #Condition 2:
            #if node is an edge node, always go one node closer to cloud
            elif e_length == 1:
                #print("its an edge node, move closer to cloud")
                child = self.moveCloud(mutation, mutation_prob, nodes_ram, modules_ram, G, nx, random_node, random_module,
                                       random_app, currentRAM)
    
            else:
                #print("middle node so edge will always = ", e_length)
                move = random.randint(0, 1)
                if move == 0:
                    #print("its a middle node, move closer to cloud")
                    child = self.moveCloud(mutation, mutation_prob, nodes_ram, modules_ram, G, nx, random_node,
                                           random_module, random_app, currentRAM)

                else:
                    #print("its a middle node, move closer to edge")
                    child = self.moveEdge(mutation, mutation_prob, nodes_ram, modules_ram, G, nx, random_node,
                                          random_module,
                                          random_app, currentRAM)

            if child_validate == mutation:
                #print("mutation fail")
                mutation = self.newMutation(mutation, 1, nodes_ram, modules_ram, G, nx)
            #else:
                #print("mutation pass")
                #print("mutated child is", mutation)
        return mutation


    """
    Function:
        run GA
    Args:
        self: GA it self
        nodes: nodes information about the topology
        apps: apps id
        modules: 2D metric ==> app*module
        pop_size: be setted in main.py
        links: Topology.G.edges()
        nodes_ram: ram space in each nodes
        modules_ram: ram space in each modules
        msg: 3D metric ==> app*message*3
        nx: a library
        G: a graph
        bw: edge informarion, bw and pr info
        gen_size: generation size
    Returns:
        the best solution, which modules should place in which nodes, and this solution's fitness score
    """
    #bw是attPR_BW
    def run(self, nodes, apps, modules, pop_size, links, nodes_ram, modules_ram,msg,nx,G,bw,gen_size, shift_delay, old_placement, sim_parameters, updated_monitored_latency_edge,old_plan_measured_latency_workflow, interval, t, static_flag, sender_msg_size):

        #create init population
        
        population = self.initialization(nodes, apps, modules, pop_size, links, nodes_ram, modules_ram,nx,G)

        #print(population)
        
        population_fit = []
        population_individual = []
        population_static_latency = []
        population_adaptation_time=[]
        population_latency=[]
        population_estimated_latency=[]

        best_fit = 0

        converge = 0 #收敛
        threshold = gen_size
        

        #instead of using a threshold, stop the code when the algorithm does really converge
        #相当于每一次迭代
        while converge < threshold:
            nextGeneration = []

            #QUESTION: add another loop here in one note? why
            #print("size of population is", len(population))
            #print("\n\ngeneration number",converge)
            #print("the population is")
            #for i in population:
                #print(i)

            #len(population) is pop_size
            for i in range(0,len(population)):
                    
                o_1, o_2 = self.select(population, nodes_ram, modules_ram, msg, G, bw, nx, shift_delay, old_placement, sim_parameters,updated_monitored_latency_edge, old_plan_measured_latency_workflow, interval, t, False, static_flag, sender_msg_size)
                #o_1 is father
                #0_2 is mother

                #print("\n\nfather", o_1)
                #print("mother", o_2)
                #print("_______________________________________________")

                child, mutation_prob = self.crossover(o_1, o_2, nodes, nodes_ram, modules, modules_ram,nx,G)
                #print("child", child)

                #print("mutation probability", mutation_prob)
                #mutated_child = self.mutate(child,mutation_prob,nodes_ram, modules_ram,G, nx)
                mutated_child = self.newMutation(child, mutation_prob, nodes_ram, modules_ram, G, nx)

                nextGeneration.append(mutated_child)



            population = []
            population = copy.deepcopy(nextGeneration)
            #print("最终population：",population)
            
            #fitness = self.population_fitness(population, nodes_ram, modules_ram, msg, G,bw,nx)
            #fitness, individual, latency, estimated_latency = self.select(population, nodes_ram, modules_ram, msg, G,bw,nx,shift_delay, old_placement, sim_parameters, True)
            if static_flag==False:
                fitness, individual, static_latency, adaptation_time = self.select(population, nodes_ram, modules_ram, msg, G,bw,nx,shift_delay, old_placement, sim_parameters,updated_monitored_latency_edge,old_plan_measured_latency_workflow, interval,t,  True, static_flag, sender_msg_size)
                population_fit.append(fitness)
                population_individual.append(individual)
                population_static_latency.append(static_latency)
                population_adaptation_time.append(adaptation_time)
            else:
                fitness, individual=self.select(population, nodes_ram, modules_ram, msg, G,bw,nx,shift_delay, old_placement, sim_parameters,updated_monitored_latency_edge,old_plan_measured_latency_workflow, interval,t,  True, static_flag, sender_msg_size)
                population_fit.append(fitness)
                population_individual.append(individual)
                
            

            #population_latency.append(latency)
            #population_estimated_latency.append(estimated_latency)
            converge +=1


        #print("best individual is", idx, population[idx])

        if static_flag==False:
            best_fit = np.max(population_fit)
            #print("max is", population_fit)
            val, idx = max((val, idx) for (idx, val) in enumerate(population_fit))
        else:
            best_fit = np.min(population_fit)
            val, idx = min((val, idx) for (idx, val) in enumerate(population_fit))


        
        #print("best fitness score:", population_fit[idx])
        #print("best latency:",population_latency[idx],population_estimated_latency[idx])
        #print("fitness_scores_list:", population_fit)
        #print("population_measured_latency:",population_latency)
        #print("population_estimated_latency:", population_estimated_latency)
        
        #print("\nGA solution")
        #print(population_individual[idx])
        #print(population_individual[idx-1])
        #val = 1/val
        #print("fitness: %.4f" % best_fit)
        #print("static latency:", population_static_latency[idx])
        #print("adaptation cost:",population_adaptation_time[idx] )
        #print("measured latency:", population_latency[idx])
        #print("estimated latency:", population_estimated_latency[idx])
        


        #print("population fitness is", population_fit)
        #print("best fitness is", best_fit)
        #multiplied_list = [element * 10000 for element in population_fit]
        #population_fit.reverse()


        #return population_individual[idx], val, population_latency[idx], population_estimated_latency[idx]
        if static_flag==False:
            return population_individual[idx], val, population_static_latency[idx], population_adaptation_time[idx]
        else:
            return population_individual[idx], val
        #print("GA converge\n\n ###############################################################")