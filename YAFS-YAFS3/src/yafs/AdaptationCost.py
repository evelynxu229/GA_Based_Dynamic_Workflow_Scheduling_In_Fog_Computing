import networkx as nx
import sys
sys.path.insert(0,'/Users/tingxu/Desktop/Fog-computing/Fog-Computing-Dissertation-Code/YAFS-YAFS3/YAFS-YAFS3/src/yafs/')
#print(sys.path)

class AdaptationCostCalculator:
    def __init__(self, placement_1, placement_2, G, attBW_PR, MSG):
        self.placement_1 = placement_1
        self.placement_2 = placement_2
        self.G = G
        self.attBW_PR = attBW_PR
        self.MSG = MSG

    def find_msg_size_in_edge(self, app_module_index):
        
        msg_size_sum = 0
        for msg_app in self.MSG:
            for msg in msg_app:
                if msg[0] == app_module_index or msg[1] == app_module_index:
                    msg_size_sum += msg[2]
        return msg_size_sum

    def calculate_moving_node_cost(self, edge_start, edge_end, app_module_index):
        
        try:
            bw_att = self.attBW_PR[(edge_start, edge_end)]
        except KeyError:
            try:
                bw_att = self.attBW_PR[(edge_end, edge_start)]
            except KeyError:
                bw_att = None  # 或者你可以选择抛出自定义的异常，或者记录错误
        try:
            pr_att = self.attBW_PR[(edge_start, edge_end)]
        except KeyError:
            try:
                pr_att = self.attBW_PR[(edge_end, edge_start)]
            except KeyError:
                pr_att = None  # 或者你可以选择抛出自定义的异常，或者记录错误
        #bw_att = self.attBW_PR.get((edge_start, edge_end))
        #pr_att = self.attBW_PR.get((edge_start, edge_end))
        code_size = 10*1024
        #print("calculate_moving_node_cost:", msg_size / bw_att + pr_att)
        return code_size /(1000000.0* bw_att) + pr_att

    def get_path(self, node_1, node_2):
        shortest_path = nx.shortest_path(self.G, node_1, node_2)
        return shortest_path

    def get_corresponding_component(self, com_1_index_x, com_1_index_y):
        return self.placement_1[com_1_index_x][com_1_index_y]

    def calculate(self):

        #print("old_placement:", self.placement_1)
        #print("new_placement:", self.placement_2)
        cost = 0
        app_index = 0
        for each_app in self.placement_2:
            module_index = 0
            first_index = {}
            for i, num in enumerate(each_app):
                if num not in first_index:
                    first_index[num] = i
            list_without_duplicates = list(first_index.keys())
            #print("each app:", each_app)
            #print("list_without_duplicates:", list_without_duplicates)
            #print("dict_index:", first_index)
            #component_1是新的
            for component_1 in list_without_duplicates:
                #component_2是新的
                component_2 = self.get_corresponding_component(app_index, first_index[component_1])
                #print("component1:", component_1)
                #print("component2:", component_2)
                path = self.get_path(component_1, component_2)
                #print("adapt path:",component_1, component_2, path)
                
                #print("path from component1 to component2:",path)
                for i in range(0, len(path) - 1):
                    app_module_index = str(app_index) + "_0" + str(module_index + 1)
                    cost = cost + self.calculate_moving_node_cost(path[i], path[i + 1], app_module_index)
                module_index += 1
            app_index += 1
        return cost
