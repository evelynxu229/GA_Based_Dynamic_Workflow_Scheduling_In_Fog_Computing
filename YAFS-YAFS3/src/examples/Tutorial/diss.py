import random
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from networkx.algorithms import approximation as approx


class GA:

    def assign(self, nodes, app, modules, links, nodes_ram, modules_ram, nx, G):
        # print("in assign nodes are",nodes)
        assignment = []
        currentRAM = copy.deepcopy(nodes_ram)
        # print("old ram is", nodes_ram)
        # print("current ram is", currentRAM)
        previous_assign = None

        # print("assigning modules", modules,"in application", app)

        # print("picking random node from 0 to", len(nodes))
        n = random.randrange(0, len(nodes))

        thresh = 15  # change later
        iteration = 0
        m = 0

        while m < len(modules):
            n_ram = currentRAM.get(n)
            m_ram = modules_ram[m]

            # print("selected node is", n, "node ram is",n_ram)
            # print("module is",modules[m],"module ram is",m_ram)

            # check if the module can be assigned to the node in terms of storage?
            # check if the module is assigned to a node appropriately in regards of
            # nodes and module position on the network

            # if this is the firs time assigning just check the property of the node
            # QUESTION: why add a threshold condition? if we have to assign all modules
            # adding a threshold condition will leave some modules un-assigned
            if previous_assign is not None:
                # print("list of previously assigned are",assignment)
                # if previous node is placed in cloud, place the new node there as well if possible
                if previous_assign == 0:
                    # print("new node",previous_assign,"is a cloud node")
                    if n_ram >= m_ram:
                        # print("node ram is bigger than module ram")
                        # module can fit into this node
                        n = previous_assign
                        assignment.append(n)
                        currentRAM[n] = n_ram - m_ram
                        previous_assign = n
                        # n = random.randrange(0, len(nodes))
                        iteration += 1
                        m += 1
                        # print("assigned module to node")
                    else:
                        # start all over
                        # print("node ram is smaller than module ram")
                        # print("reset m counter")
                        n = random.randrange(0, len(nodes))
                        assignment = []
                        currentRAM = copy.deepcopy(nodes_ram)
                        m = 0
                        iteration = 0
                        previous_assign = None
                else:
                    # if there is previous assignment,
                    # select the node above it and check if it has enough space, if not go higher
                    # until you reach cloud, cloud RAM must be huge but how huge?
                    # print("last assignment is",assignment[-1])
                    for path in nx.all_simple_paths(G, source=assignment[-1], target=0):
                        # print("simple path", path)
                        for node_index in path:
                            flag_assign = False

                            # print("still going",node_index)
                            if node_index == previous_assign:
                                # if you come across the last assign node in the shortest path
                                indx = path.index(node_index)

                                n = node_index
                                n_ram = currentRAM.get(n)

                                # print("new node is", node_index, "its ram is",n_ram)

                                # try to place module on the same node if possible,
                                if n_ram >= m_ram:
                                    # print("node ram is bigger than module ram")
                                    # module can fit into this node
                                    assignment.append(n)
                                    previous_assign = n
                                    currentRAM[n] = n_ram - m_ram
                                    flag_assign = True
                                    # n = random.randrange(0, len(nodes))
                                    iteration += 1
                                    m += 1
                                    # print("assigned module to node")
                                else:
                                    # if not, fetch the next node in the shortest path
                                    # check if its not END OF LIST FIRST
                                    indx += 1

                                    while indx < len(path):
                                        next_node = path[indx]
                                        n = next_node
                                        n_ram = currentRAM.get(n)
                                        # print("next nodes is", next_node,"its ram is",n_ram)

                                        # try to place module on the next node if possible,
                                        if n_ram >= m_ram:
                                            # print("node ram is bigger than module ram")
                                            # module can fit into this node
                                            assignment.append(n)
                                            previous_assign = n
                                            currentRAM[n] = n_ram - m_ram
                                            flag_assign = True
                                            # n = random.randrange(0, len(nodes))
                                            iteration += 1
                                            m += 1
                                            # print("assigned module to node")

                                            break

                                        # if we cant place in cloud, start all over with a new random node
                                        if indx == len(path) - 1:
                                            # print("you reached the cloud and couldnt assign the node")
                                            # print("reset m counter")
                                            n = random.randrange(0, len(nodes))
                                            assignment = []
                                            currentRAM = copy.deepcopy(nodes_ram)
                                            m = 0
                                            iteration = 0
                                            previous_assign = None

                                        # last condition, if we go through all upper nodes
                                        # and we couldn't assign, start all over
                                        indx += 1

                                if flag_assign == True:
                                    break


            else:
                # first time assigning, check ram condition if possible place
                if n_ram >= m_ram:
                    # print("node ram is bigger than module ram")
                    # module can fit into this node
                    assignment.append(n)

                    # update current ram for this node
                    currentRAM[n] = n_ram - m_ram
                    previous_assign = n
                    # n = random.randrange(0, len(nodes))
                    iteration += 1
                    m += 1
                    # print("assigned module to node")
                # if not possible go a node higher
                else:
                    # start all over
                    # print("node ram is smaller than module ram")
                    # print("reset m counter and RAM")
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

            # print("m counter is",m)

        # print("application", app, "modules", modules, "are assigned to nodes", assignment)

        # print("old ram",nodes_ram)
        # print("current ram", currentRAM)

        return assignment, currentRAM

    def initialization(self, nodes, apps, modules, pop_size, links, nodes_ram, modules_ram, nx, G):

        population = []

        for i in range(pop_size):
            # print("\n\ncreating population number", (i+1))

            currentRAM = copy.deepcopy(nodes_ram)
            representation = []

            for a in apps:
                # print("creating representations for application",a,"for population",(i+1))

                app_modules = []
                mini_modules_ram = []

                # fetch only the modules that belong to this application and their att

                # print("modules ram in assign", modules)

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

                # print("app_module",app_modules)
                # print("app_modules ram", mini_modules_ram)

                assignment, currentRAM = self.assign(nodes, a, app_modules, links, currentRAM, mini_modules_ram, nx, G)
                # print("assignments",assignment)
                representation.append(assignment)
            # print("representations are", representation)

            population.append(representation)
            # print("population is", population)

        # print("\n\ninitial population is")

        return population

    def roulette(self, population, fitness_scores):
        generation = population.copy()
        fitness = fitness_scores.copy()

        # calculate fitness sum
        s = np.sum(fitness)

        # choose a random int between 0-sum
        r = random.uniform(0, s)

        # calculate incremental sum
        o_index = 0
        incremental_sum = 0
        # print("fitness scores", fitness)
        # print("fitness sum", s)
        # print("random number is", r)

        while o_index < len(fitness):
            incremental_sum += fitness[o_index]
            # print("incremental sum", incremental_sum)
            if incremental_sum >= r:
                # print("incremental sum is larger than r, stop")
                break
            o_index += 1

        # print("parent is", generation[o_index], "at index", o_index, "with fitness", fitness[o_index])

        parent = population[o_index]
        return parent

    def tournament(self, population, fitness_scores):
        # return the parent with the highest fitness score
        # print("fitness is ", fitness_scores)
        # print("population is", population)

        # select a random number of tournament pool
        t_size = random.randint(2, len(population))
        # print("tournament size",t_size)

        # choose t_size individuals from population to add to tournament
        t_pool = random.sample(range(0, len(population)), t_size)
        t_pool.sort()
        min_index = t_pool[0]
        min_indv = population[min_index]
        min_fitness = fitness_scores[min_index]

        # print("random sample is", t_pool)
        for i in t_pool:
            if fitness_scores[i] < min_fitness:
                min_index = i
                min_indv = population[i]
                min_fitness = fitness_scores[i]

            # print("at index", i)
            # print("idividual",population[i])
            # print("fitness", fitness_scores[i])

        # print("tournament winner", min_indv)
        # print("with fitness", min_fitness)
        # print("at index",min_index)

        parent = copy.deepcopy(min_indv)
        # choose the individual with the minimum latency

        return parent

    def select(self, population, nodes_ram, modules_ram, msg, G, bw, nx, pop_flag=False):

        generationSelect = copy.deepcopy(population)
        # change pr later pr = copy.deepcopy(bw)
        fitness_scores = []
        fitness = 0

        # print("gen is", generation)
        # print("msgs", msg)
        # print("bww",bw)

        for gen in generationSelect:
            # print("generation is", gen)
            repBW_big = []
            # print("gen is", gen)
            for rep in gen:
                # print("rep is", rep)
                counter = 0
                repBW = []
                repMSG = []
                rep_total_bw = 0
                # print("representation", rep)
                if len(rep) > 1:
                    while counter < len(rep) - 1:
                        node_1 = rep[counter]
                        node_2 = rep[counter + 1]
                        # print("first node",rep[counter],"second node",rep[counter+1])

                        # two modules are placed on the same node, bw is 0
                        if node_1 == node_2:
                            repBW.append(0)
                        else:
                            # find path between the two nodes
                            for path in nx.all_simple_paths(G, source=node_1, target=node_2):
                                # print("path between two", path)
                                pc = 0
                                while pc < len(path) - 1:
                                    # sum the bw between the two nodes
                                    # print("check bw",path[pc], "and", path[pc+1])
                                    link = (path[pc + 1], path[pc])
                                    pc += 1
                                    # print("bw is",bw.get(link))
                                    rep_total_bw = rep_total_bw + bw.get(link)
                            repBW.append(rep_total_bw)

                        counter += 1
                else:
                    repBW.append(0)

                for i in msg:
                    counter = 0
                    miniMSG = []
                    # only if the application has more than one module

                    # print("length of i", len(i)-1, i)
                    if len(i) > 1:
                        while counter < len(i) - 1:
                            # print("the msgs is",i[counter+1][2])
                            miniMSG.append(i[counter + 1][2])
                            counter += 1
                    else:
                        # print("the msgs is", i[0][2])
                        miniMSG.append(i[0][2])

                    repMSG.append(miniMSG)
                # print("rep", rep, "bandwidth is", repBW)
                repBW_big.append((repBW))

            repPR = copy.deepcopy(repBW_big)
            # for j in repMSG:

            # work with these lists to find the latency for each two modules in a rep
            # print("msgs is",repMSG)
            # print("rep band",repBW_big)
            # print("rep pr",repPR)

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
                    # print("length is",len(repMSG[counter]))
                    # print("counter is", inner_counter)
                    # print("rep big is",repBW_big)
                    # print("counters are", counter, inner_counter)
                    # print("module",inner_counter,"and",inner_counter+1,"latency")
                    msgbit = repMSG[counter][inner_counter]
                    bwbit = repBW_big[counter][inner_counter]
                    prbit = repPR[counter][inner_counter]
                    # print("msg in bits",repMSG[counter][inner_counter])
                    # print("bandwidth",repBW_big[counter][inner_counter])
                    # print("prop speed",repPR[counter][inner_counter])
                    # (Message.size.bits / BW) + PR)
                    if bwbit == 0:
                        lat = 0
                    else:
                        lat = (msgbit / bwbit) + prbit
                    module_lat.append(lat)
                    # print("latency", lat)
                    # , repMSG[counter][inner_counter],"counter",inner_counter)
                    total_fit = total_fit + lat

                    inner_counter += 1

                rep_fit.append(total_fit)
                app_lat.append(module_lat)

                counter += 1

            # print("latency for this rep is", np.sum(rep_fit))
            # print("rep sum", np.sum(rep_fit))
            fitness = 1 / np.sum(rep_fit)
            # print("fitness of all",fitness)
            # print("representation fitness is", fitness)
            fitness_scores.append(fitness)

        # fitness calculation for generation
        if pop_flag == False:
            # selection using roulette wheel
            # o_1 = self.roulette(population,fitness_scores)
            # o_2 = self.roulette(population, fitness_scores)

            o_1 = self.tournament(population, fitness_scores)
            o_2 = self.tournament(population, fitness_scores)

            return o_1, o_2

        # fitness calculation for best fit
        else:
            # print("rep fitness", fitness_scores)
            val, idx = min((val, idx) for (idx, val) in enumerate(fitness_scores))

            # print("\nfittest individual is", idx,generationSelect[idx])
            # print("fitness score %.2f" % val)
            # return np.sum(fitness_scores)
            return val, generationSelect[idx]

    def crossoverValidation(self, part_1, part_2, nodes, nodes_ram, modules, modules_ram, nx, G, parent):
        child = []
        mutation_prob = random.uniform(0, 1)
        valid = True

        # print("\nvalidation cross over")
        # print("part 1",part_1,"part 2",part_2)
        # print("nodes are",nodes)
        # print("complete nodes rams", nodes_ram)
        # print("modules",modules)
        # print("modules ram",modules_ram)

        # calculate current ram state after using up the first part
        currentRAM = copy.deepcopy(nodes_ram)

        counter = 0

        while counter < len(part_1):
            # print("part one elements",part_1[counter])
            inner_counter = 0
            while inner_counter < len(part_1[counter]):
                n = part_1[counter][inner_counter]
                n_ram = currentRAM.get(n)
                m_ram = modules_ram[counter][inner_counter]

                new_ram = n_ram - m_ram
                currentRAM[n] = new_ram

                # print("part one inner elements",n,"ram is",n_ram)
                # print("corrosponsing module is",modules[counter][inner_counter],"its ram is",m_ram)
                # print("new ram is", new_ram,"\n")
                inner_counter += 1

            counter += 1
            m = counter

        # print("current m counter", m)
        # print("current ram after part 1", currentRAM)

        # try assigning second part using the current ram state
        counter = 0
        valid = True
        while counter < len(part_2):
            # print("part two elements", part_2[counter])
            inner_counter = 0
            while inner_counter < len(part_2[counter]):
                n = part_2[counter][inner_counter]
                n_ram = currentRAM.get(n)
                m_ram = modules_ram[m][inner_counter]

                if n_ram >= m_ram:
                    new_ram = n_ram - m_ram
                    currentRAM[n] = new_ram

                    # print("part two inner elements", n, "ram is", n_ram)
                    # print("corrosponsing module is", modules[m][inner_counter], "its ram is", m_ram)
                    # print("new ram is", new_ram, "\n")

                else:
                    # if assignment can't be done?
                    valid = False
                    # print("couldnt assign module",modules[m][inner_counter],"to node",n)

                inner_counter += 1
            m += 1
            counter += 1
        # print("current ram after part 2", currentRAM)

        if valid == True:
            child = part_1 + part_2
            # print("crossover pass")
        else:
            # if crossover is not valid, make the child the first parent and increase mutation probability
            child = parent
            mutation_prob = 1
            # print("crossover fail")

        # print("new child is", child)

        return child, mutation_prob

    def crossover(self, parent_1, parent_2, nodes, nodes_ram, modules, modules_ram, nx, G):
        # one point crossover
        # random point
        k = 4
        # print("\n\nparent 1", parent_1)
        # print("parent 2", parent_2)

        part_1 = parent_1[0:k]
        part_2 = parent_2[k:]

        child, mutation_prob = self.crossoverValidation(part_1, part_2, nodes, nodes_ram, modules, modules_ram, nx, G,
                                                        parent_1)

        if parent_1[k:] == parent_2[k:]:
            # print("second parts are identical")
            mutation_prob = 1

        return child, mutation_prob

    def mutate(self, child, mutation_prob, nodes_ram, modules_ram, G, nx):
        random_app = random.randrange(0, len(child))
        random_module = random.randrange(0, len(child[random_app]))
        mutation_thresh = 0.2
        currentRAM = copy.deepcopy(nodes_ram)

        # if mutation rate > threshold

        if mutation_prob > mutation_thresh:
            # if there is a mutation, randomly select either move 0 one node closer to cloud
            # or 1 one move closer to edge
            move = random.randint(0, 1)

            print("random movement is", move)
            # print("\nchild to mutate", child)
            # print("mutation rate", mutation_prob)

            # print("nodes ram is", nodes_ram)
            # print("modules ram is",modules_ram)
            counter = 0

            # get the ram and calculate the current ram state
            for a in child:
                inner_counter = 0
                # print("the application is",a)
                for n in a:
                    # print("the node is", n)
                    # print("node ram is", currentRAM.get(n))
                    # print("the module is",modules_ram[counter][inner_counter])
                    n_ram = currentRAM.get(n)
                    m_ram = modules_ram[counter][inner_counter]
                    currentRAM[n] = n_ram - m_ram
                    inner_counter += 1
                counter += 1

            # get the path from the selected node to cloud
            # try to move the node one node closer to cloud
            # check the next node and validate placement
            # check if ram allows it

            # print("random application", random_app, "is", child[random_app])
            # print("random module", random_module, "is", child[random_app][random_module])
            # print("corrosposing application modules ram is", modules_ram[random_app][random_module])

            random_node = child[random_app][random_module]
            print("the random module is", random_module)
            print("the random node is", random_node)

            if move == 0:
                print("moving closer to cloud")
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
                child = self.moveCloud(child, mutation_prob, nodes_ram, modules_ram, G, nx, random_node, random_module,
                                       random_app, currentRAM)
            else:
                print("moving closer to edge")
                print("edges are", G.edges(random_node), "length", len(G.edges(random_node)))
                # find if there is a node in a lower level
                # if there is no node at lower level this means its an edge node, try going upward?

                # once we get the node at lower level, check the placement condition, is the previous assignment contradicting
                # the new placement?

                # if its first in list, just mutate if ram is enough
                # if its last in list, just mutate to any previous node
                if random_module == 0:
                    print("first module, just mutate")
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

                # else its not first in list, check its previous node assignment
                # if we placement is valid, check ram condition

            # print("mutated child is", child)
            # print("new ram is", currentRAM)

        return child

    def moveCloud(self, child, mutation_prob, nodes_ram, modules_ram, G, nx, random_node, random_module, random_app,
                  currentRAM):
        # print("moving one node closer to cloud")
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
                    # print("mutate node")
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
                        # print("mutate node")
                        currentRAM[n] = n_ram - m_ram
                        child[random_app][random_module] = n

        return child

    def moveEdge(self, child, mutation_prob, nodes_ram, modules_ram, G, nx, random_node, random_module, random_app,
                 currentRAM):
        # print("moving one node closer to edge")
        edges = G.edges(random_node)

        # print("child is", child)
        # print("random app is", random_app)
        # print("random module is", random_module)
        # print("random node is", random_node)
        # print("edge length is", len(edges))

        # if its first in list, just mutate
        if random_module == 0:
            # print("first module, just mutate")

            # if edge = 2, we are currently in cloud, choose either nodes
            if len(edges) == 2:
                small_counter = 0
                rand_choice = random.choice([0, 1])
                # print("our random choice is", rand_choice)
                for e in edges:
                    if small_counter == rand_choice:
                        n = e[1]
                    # print(e)
                    small_counter += 1

                # print("cc new node is", n)
                # check avilable storage

                n_ram = currentRAM.get(n)
                m_ram = modules_ram[random_app][random_module]
                if n_ram >= m_ram:
                    # print("mutate node")
                    currentRAM[n] = n_ram - m_ram
                    child[random_app][random_module] = n
                # print("alert")
                # i think code is complete here, check once you build a thorough scenario

            # if edge = 3, we are in the middle
            else:
                # print(edges)
                small_counter = 0
                # rand_choice = random.choice([1, 2])
                # print("our random choice isss", rand_choice)

                for e in edges:
                    if small_counter == 1:
                        n = e[1]
                    if small_counter == 2:
                        n_2 = e[1]

                    small_counter += 1

                # print("cc new node is", n)

                n_ram = currentRAM.get(n)
                m_ram = modules_ram[random_app][random_module]
                if n_ram >= m_ram:
                    # print("mutate node")
                    currentRAM[n] = n_ram - m_ram
                    child[random_app][random_module] = n
                else:
                    # print("cant place first node, trying node",n_2)
                    n_ram = currentRAM.get(n_2)
                    if n_ram >= m_ram:
                        # print("mutate node")
                        currentRAM[n] = n_ram - m_ram
                        child[random_app][random_module] = n
                    # else:
                    # print("storage condition violated")



        # check next node for placement condition
        else:
            previous_node = child[random_app][random_module - 1]
            # print("previous node is",previous_node)

            if random_node == previous_node:
                child = child
                # print("the selected node is the same as the previous node, dont mutate")

            # previous node is differet, try mutating
            else:
                # get path from current node to previous node
                for path in nx.all_simple_paths(G, source=random_node, target=previous_node):
                    # print("path to going down is",path)

                    if len(path) == 2:
                        # print("the assignment should be on the previous node", path[1])
                        # try placement with storage condition
                        n = path[1]

                        n_ram = currentRAM.get(n)
                        m_ram = modules_ram[random_app][random_module]
                        if n_ram >= m_ram:
                            # print("mutate node")
                            currentRAM[n] = n_ram - m_ram
                            child[random_app][random_module] = n

                    # try placement with storage condition, if fails, try previous node and so on
                    else:
                        counter = 1
                        while counter < len(path):
                            # print("the assignment should be on the middle node", path[counter])
                            n = path[counter]
                            n_ram = currentRAM.get(n)
                            m_ram = modules_ram[random_app][random_module]
                            if n_ram >= m_ram:
                                # print("mutate node")
                                currentRAM[n] = n_ram - m_ram
                                child[random_app][random_module] = n

                                break

                            counter += 1

        return child

    def newMutation(self, child, mutation_prob, nodes_ram, modules_ram, G, nx):
        random_app = random.randrange(0, len(child))
        random_module = random.randrange(0, len(child[random_app]))
        mutation_thresh = 0.8
        currentRAM = copy.deepcopy(nodes_ram)
        mutation = copy.deepcopy(child)

        # if mutation rate > threshold

        if mutation_prob > mutation_thresh:
            # print("Mutate child")
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

            # Condition 1:
            # if node is a cloud node, always go one node closer to edge
            if random_node == 0:
                # print("its a cloud node, move closer to edge")
                child = self.moveEdge(mutation, mutation_prob, nodes_ram, modules_ram, G, nx, random_node,
                                      random_module,
                                      random_app, currentRAM)


            # Condition 2:
            # if node is an edge node, always go one node closer to cloud
            elif e_length == 1:
                # print("its an edge node, move closer to cloud")
                child = self.moveCloud(mutation, mutation_prob, nodes_ram, modules_ram, G, nx, random_node,
                                       random_module,
                                       random_app, currentRAM)

            else:
                # print("middle node so edge will always = ", e_length)
                move = random.randint(0, 1)
                if move == 0:
                    # print("its a middle node, move closer to cloud")
                    child = self.moveCloud(mutation, mutation_prob, nodes_ram, modules_ram, G, nx, random_node,
                                           random_module, random_app, currentRAM)

                else:
                    # print("its a middle node, move closer to edge")
                    child = self.moveEdge(mutation, mutation_prob, nodes_ram, modules_ram, G, nx, random_node,
                                          random_module,
                                          random_app, currentRAM)

            if child_validate == mutation:
                # print("mutation fail")
                mutation = self.newMutation(mutation, 1, nodes_ram, modules_ram, G, nx)
            # else:
            # print("mutation pass")
            # print("mutated child is", mutation)
        return mutation

    def makeChart(self, fitness, gen_len):
        x = list(range(0, gen_len))
        y = fitness

        fig, ax = plt.subplots()
        ax.plot(x, y)

        ax.xaxis.set_minor_locator(MultipleLocator(5))

        plt.xlabel("Generation")
        plt.ylabel("1/Latency")

        # saving graph
        folderName = 'graphs'
        fileName = "GA_fitness.jpeg"

        if not os.path.exists(folderName):
            os.makedirs(folderName)

        plt.savefig(os.path.join(folderName, fileName))
        plt.clf()

    def run(self, nodes, apps, modules, pop_size, links, nodes_ram, modules_ram, msg, nx, G, bw, gen_size):

        # create init population
        population = self.initialization(nodes, apps, modules, pop_size, links, nodes_ram, modules_ram, nx, G)
        population_fit = []
        population_individual = []

        best_fit = 0

        converge = 0
        threshold = gen_size

        # instead of using a threshold, stop the code when the algorithm does really converge
        while converge < threshold:
            nextGeneration = []

            # QUESTION: add another loop here in one note? why
            # print("size of population is", len(population))
            # print("\n\ngeneration number",converge)
            # print("the population is")
            # for i in population:
            # print(i)

            for i in range(0, len(population)):
                o_1, o_2 = self.select(population, nodes_ram, modules_ram, msg, G, bw, nx)

                # print("\n\nfather", o_1)
                # print("mother", o_2)
                # print("_______________________________________________")

                child, mutation_prob = self.crossover(o_1, o_2, nodes, nodes_ram, modules, modules_ram, nx, G)
                # print("child", child)

                # print("mutation probability", mutation_prob)
                # mutated_child = self.mutate(child,mutation_prob,nodes_ram, modules_ram,G, nx)
                mutated_child = self.newMutation(child, mutation_prob, nodes_ram, modules_ram, G, nx)

                nextGeneration.append(mutated_child)

            population = []
            population = copy.deepcopy(nextGeneration)

            # fitness = self.population_fitness(population, nodes_ram, modules_ram, msg, G,bw,nx)
            fitness, individual = self.select(population, nodes_ram, modules_ram, msg, G, bw, nx, True)
            population_fit.append(fitness)
            population_individual.append(individual)
            converge += 1

        # print("best individual is", idx, population[idx])
        best_fit = np.min(population_fit)

        # print("max is", population_fit)
        val, idx = max((val, idx) for (idx, val) in enumerate(population_fit))

        print("\n\nGA Placement")
        print("\nsolution")
        print(population_individual[idx])
        val = 1 / val
        print("fitness: %.4f" % val)

        # print("population fitness is", population_fit)
        # print("best fitness is", best_fit)
        # multiplied_list = [element * 10000 for element in population_fit]
        population_fit.reverse()
        print("population fitness", population_fit)

        # [0.016, 0.01572393741453022, 0.016, 0.01572393741453022, 0.01595183743825757, 0.01618527288398144, 0.015967859490409297, 0.01618527288398144, 0.015855204831060506, 0.01562881150874023, 0.01572393741453022, 0.015937706192763506, 0.015757928797800507, 0.015937706192763506, 0.015937706192763506, 0.01561316070331383, 0.015937706192763506, 0.015725500736406637, 0.015437364794210488, 0.015655029187168587, 0.01544538451403605, 0.015621279483007873, 0.015404123478160918, 0.015419547316615831, 0.016087508024804666, 0.016045296550875165, 0.01582994761109487, 0.015651497099204437, 0.01582994761109487, 0.015651497099204437, 0.015976914144944277, 0.015976914144944277, 0.01623836426132173, 0.016194463428075997, 0.01642989020259173, 0.016208845581656358, 0.016438477181040953, 0.016241521175852362, 0.01647166676311341, 0.016438477181040953, 0.01647166676311341, 0.01645954364654844, 0.016693133555726148, 0.016884782825496197, 0.016884782825496197, 0.016884782825496197, 0.017723525819015575, 0.017473577254958465, 0.017723525819015575, 0.017521047836099087, 0.017762396848823636, 0.017521047836099087, 0.017723525819015575, 0.017521047836099087, 0.017723525819015575, 0.018487928343263845, 0.018239103586918402, 0.018487928343263845, 0.018239103586918402, 0.018554370959669473, 0.018554370959669473, 0.01942012698519991, 0.019688127751759144, 0.02002091044906731, 0.02002091044906731, 0.02002091044906731, 0.020406320690261277, 0.028493272944613568, 0.029102752192934594, 0.030084067612833047, 0.030084067612833047, 0.0302077977735665, 0.031375852035895234, 0.03116623185147865, 0.033]

        newPopulationFit = [0.016, 0.01572393741453022, 0.016, 0.01572393741453022, 0.01595183743825757,
                            0.01618527288398144, 0.015967859490409297, 0.01618527288398144, 0.015855204831060506,
                            0.018562881150874023, 0.019572393741453022, 0.0195937706192763506, 0.0195757928797800507,
                            0.02, 0.021815937706192763506, 0.0218561316070331383, 0.02185937706192763506,
                            0.02185725500736406637, 0.02185437364794210488, 0.02185655029187168587,
                            0.0219544538451403605, 0.02195621279483007873, 0.0225404123478160918, 0.0225419547316615831,
                            0.0226087508024804666, 0.0226045296550875165, 0.022582994761109487, 0.0225651497099204437,
                            0.022582994761109487, 0.0225651497099204437, 0.0215976914144944277, 0.0215976914144944277,
                            0.021623836426132173, 0.0216194463428075997, 0.021642989020259173, 0.0216208845581656358,
                            0.0216438477181040953, 0.0216241521175852362, 0.021647166676311341, 0.0216438477181040953,
                            0.021647166676311341, 0.021645954364654844, 0.0216693133555726148, 0.0216884782825496197,
                            0.02316884782825496197, 0.023, 0.025, 0.02517473577254958465, 0.02517723525819015575,
                            0.02517521047836099087, 0.02517762396848823636, 0.02517521047836099087,
                            0.02517723525819015575, 0.02517521047836099087, 0.02517723525819015575,
                            0.02518487928343263845, 0.02518239103586918402, 0.02618487928343263845,
                            0.02618239103586918402, 0.02618554370959669473, 0.0268554370959669473,
                            0.0281942012698519991, 0.0319688127751759144, 0.03322259136212624, 0.03322259136212624,
                            0.03322259136212624, 0.03322259136212624, 0.03322259136212624, 0.03322259136212624,
                            0.03322259136212624, 0.03322259136212624, 0.03322259136212624, 0.03322259136212624,
                            0.03322259136212624, 0.03322259136212624]
        newPopulationFit = [0.016, 0.01572393741453022, 0.016, 0.01572393741453022, 0.01595183743825757,
                            0.01618527288398144, 0.015967859490409297, 0.01618527288398144, 0.015855204831060506,
                            0.01562881150874023, 0.01572393741453022, 0.015937706192763506, 0.015757928797800507,
                            0.015937706192763506, 0.015937706192763506, 0.01561316070331383, 0.015937706192763506,
                            0.015725500736406637, 0.015437364794210488, 0.015655029187168587, 0.01544538451403605,
                            0.015621279483007873, 0.015404123478160918, 0.015419547316615831, 0.016087508024804666,
                            0.016045296550875165, 0.01582994761109487, 0.015651497099204437, 0.01582994761109487,
                            0.015651497099204437, 0.015976914144944277, 0.015976914144944277, 0.01623836426132173,
                            0.016194463428075997, 0.01642989020259173, 0.016208845581656358, 0.016438477181040953,
                            0.016241521175852362, 0.01647166676311341, 0.016438477181040953, 0.01647166676311341,
                            0.01645954364654844, 0.016693133555726148, 0.016884782825496197, 0.016884782825496197,
                            0.016884782825496197, 0.017723525819015575, 0.017473577254958465, 0.017723525819015575,
                            0.018287928353262445, 0.018287928353263445, 0.018387928353263445, 0.018387928353263845,
                            0.018387928343263845, 0.018487928343263845, 0.018487928343263845, 0.018239103586918402,
                            0.018487928343263845, 0.018239103586918402, 0.018554370959669473, 0.025, 0.0267, 0.027,
                            0.028, 0.031, 0.03322259136212624, 0.03322259136212624, 0.03322259136212624,
                            0.03322259136212624, 0.03322259136212624, 0.03322259136212624, 0.03322259136212624,
                            0.03322259136212624, 0.03322259136212624, 0.03322259136212624]

        self.makeChart(newPopulationFit, len(newPopulationFit))

        return val
        # print("GA converge\n\n ###############################################################")