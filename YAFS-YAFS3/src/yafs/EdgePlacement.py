import random
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


class EdgePlacement:
    def fitness(self, solution, nodes_ram, modules_ram, msg, G, bw, nx):
        #print("fitness of solution")

        # within each application

        # calculate the bandwidth between each two nodes

        # calculate sum bandwidth for the application

        # change pr later pr = copy.deepcopy(bw)
        fitness = 0

        #print("generation is", gen)
        repBW_big = []
        #print("gen is", solution)
        for rep in solution:
            #print("rep is", rep)
            counter = 0
            repBW = []
            repMSG = []
            rep_total_bw = 0
            #print("representation", rep)
            if len(rep) > 1:
                while counter < len(rep)-1:
                    node_1 = rep[counter]
                    node_2 = rep[counter+1]
                    #print("first node",rep[counter],"second node",rep[counter+1])

                    # two modules are placed on the same node, bw is 0
                    if node_1 == node_2:
                        repBW.append(0)
                    else:
                        # find path between the two nodes
                        for path in nx.all_simple_paths(G, source=node_1, target=node_2):
                            #print("path between two", path)
                            pc = 0
                            while pc < len(path)-1:
                                # sum the bw between the two nodes
                                #print("check bw",path[pc], "and", path[pc+1])

                                first=min(path[pc+1], path[pc])
                                last=max(path[pc+1], path[pc])
                                link = (first, last)
                                pc += 1
                                #print("link is", link)
                                #print("bw is",bw.get(link))
                                rep_total_bw = rep_total_bw + bw.get(link)
                        repBW.append(rep_total_bw)

                    counter += 1
            else:
                repBW.append(0)

            for i in msg:
                counter = 0
                miniMSG = []
                # only if the application has more than one module

                #print("length of i", len(i)-1, i)
                if len(i) > 1:
                    while counter < len(i)-1:
                        #print("the msgs is",i[counter+1][2])
                        miniMSG.append(i[counter+1][2])
                        counter += 1
                else:
                    #print("the msgs is", i[0][2])
                    miniMSG.append(i[0][2])

                repMSG.append(miniMSG)
            #print("rep", rep, "bandwidth is", repBW)
            repBW_big.append((repBW))

        repPR = copy.deepcopy(repBW_big)
        # for j in repMSG:

        # work with these lists to find the latency for each two modules in a rep
        #print("msgs is",repMSG)
        #print("rep band",repBW_big)
        #print("rep pr",repPR)

        # latency between each two modules
        counter = 0

        app_lat = []
        rep_fit = []
        while counter < len(repMSG):
            module_lat = []
            inner_counter = 0
            total_fit = 0
            # print("application",counter,"latency")

            while inner_counter < len(repMSG[counter]):
                # print("module",inner_counter,"and",inner_counter+1,"latency")
                msgbit = repMSG[counter][inner_counter]
                bwbit = repBW_big[counter][inner_counter]
                prbit = repPR[counter][inner_counter]
                #print("msg in bits",repMSG[counter][inner_counter])
                # print("bandwidth",repBW_big[counter][inner_counter])
                #print("prop speed",repPR[counter][inner_counter])
                # (Message.size.bits / BW) + PR)
                if bwbit == 0:
                    lat = 0
                else:
                    lat = (msgbit*8/(bwbit*1000000.0)) + prbit
                module_lat.append(lat)
                #print("latency", lat)
                # , repMSG[counter][inner_counter],"counter",inner_counter)
                total_fit = total_fit + lat

                inner_counter += 1

            rep_fit.append(total_fit)
            app_lat.append(module_lat)

            counter += 1

        #print("latency for this rep is", np.sum(rep_fit))
        fitness = 1/np.sum(rep_fit)
        #print("representation fitness is", fitness)

        return fitness

    def run(self, nodes, app_list, modules, links, nodes_ram, modules_ram, msg, nx, G, bw):
        leaf_list = []

        for pair in nx.degree((G)):
            if pair[1] == 1:
                leaf_list.append(pair[0])

        a = 0
        solution = []

        for m in modules:
            assignment = []
            leaf = random.choice(leaf_list)

            module_counter = 0
            while module_counter < len(m):
                m_ram = modules_ram[a][module_counter]
                path_chosen = False
                
                # 先尝试沿着路径找
                for path in nx.all_simple_paths(G, source=leaf, target=0):
                    for n in path:
                        n_ram = nodes_ram.get(n)
                        if n_ram >= m_ram:
                            assignment.append(n)
                            leaf = n
                            nodes_ram[n] = n_ram - m_ram
                            path_chosen = True
                            break
                    if path_chosen:
                        break
                
                # 如果路径上没有合适的节点，随机选择一个
                if not path_chosen:
                    eligible_nodes = [node for node, ram in nodes_ram.items() if ram >= m_ram]
                    if eligible_nodes:
                        chosen_node = random.choice(eligible_nodes)
                        assignment.append(chosen_node)
                        nodes_ram[chosen_node] -= m_ram
                    else:
                        print("No eligible nodes found for module", m[module_counter])
                        return None  # 返回None或其他标识以表示没有找到合适的节点
                
                module_counter += 1

            solution.append(assignment)
            a += 1
        #print("solution:",solution)
        fitness = self.fitness(solution, nodes_ram, modules_ram, msg, G, bw, nx)
        return solution


    """
    def run(self,nodes, app_list, modules, links, nodes_ram, modules_ram, msg, nx, G, bw):
        #print("Edge Placement")

        #fetch all the leaves in the network
        leaf_list = []

        for pair in nx.degree((G)):
            if pair[1] == 1:
                leaf_list.append(pair[0])

        #print("leaves", leaf_list)
        #print("nodes ram", nodes_ram)


        #for every application
        a = 0
        solution = []

        for m in modules:

            assignment = []

            #print("\n\napplication number", a)
            #print("modules are",m)
            #print("module ram", modules_ram[a])

            # select a random leaf as a starting point
            leaf = random.choice(leaf_list)
            #print("first random leaf is", leaf)

            module_counter = 0
            skip = False
            while module_counter < len(m):


                #print("\nmodule", m[module_counter],"ram is",modules_ram[a][module_counter])
                m_ram = modules_ram[a][module_counter]

                #get the simple path from the leaf to the cloud
                for path in nx.all_simple_paths(G, source=leaf, target=0):

                    #print("simple path", path)
                    for n in path:
                        #fetch node ram
                        n_ram = nodes_ram.get(n)
                        #print("node",n, "its ram is",n_ram)

                        if n_ram >= m_ram:
                            assignment.append(n)
                            leaf = n
                            nodes_ram[n] = n_ram - m_ram

                            remaining = len(m) - 1 - module_counter
                            if remaining != 0 and leaf == 0:
                                skip = True
                            #print("remaining modules to assign", remaining)

                            #print("assigned module",m[module_counter],"to node",n)
                            break

                #if node is at 0, check assignment one more time because the loop wont run
                if leaf == 0 and remaining != 0:
                    if skip == False:
                        n = 0
                        n_ram = nodes_ram.get(n)
                        #print("node", n, "its ram is", n_ram)

                        if n_ram >= m_ram:
                            assignment.append(n)
                            leaf = n
                            nodes_ram[n] = n_ram - m_ram

                            remaining = len(m) - 1 - module_counter
                            #print("remaining modules to assign", remaining)

                            #print("assigned module", m[module_counter], "to node", n)

                    else:
                        skip = False

                module_counter+=1
            solution.append(assignment)
            a +=1

        #print("\n\nsolution is")
        #print(solution)

        #calculate the fitness of the solution
        fitness = self.fitness(solution, nodes_ram, modules_ram, msg, G, bw, nx)

      
        #print("\nInitial solution")
       #print(solution)
        #print("fitness:",fitness)

        return solution
    """
