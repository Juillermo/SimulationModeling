from random import random, shuffle, normalvariate, choices, choice
from itertools import combinations
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Agent:
    H = None
    L = None
    a = None

    def __init__(self, effort):
        self.payoff = None
        if effort:
            self.S = Agent.H
        else:
            self.S = Agent.L

        self.copy_S = None
        self.copy_payoff = None

        # Extension variables
        self.allocated = False
        self.node = None
        self.sum_links = None

    def receive_mark(self, mark):
        self.payoff = mark - Agent.a * self.S

    def __str__(self):
        return "S:" + str(self.S) + " payoff=" + str(self.payoff) + " node:" + str(self.node)


class Group:
    n = None
    beta = None  # weight rate of change
    gamma = None  # probability of randomly including a new member

    def __init__(self, graph=None):
        self.agents = []
        self.graph = graph
        self.links = {}

    def add_agent(self, agent):
        self.agents.append(agent)
        node = agent.node
        if node is not None:
            agent.allocated = True
            for link in self.graph[node]:
                if not self.graph.node[link].allocated:
                    weight = self.graph[node][link]["weight"]
                    if link not in self.links:
                        self.links[link] = weight
                    else:
                        self.links[link] += weight

    def give_marks(self):
        mark = sum(agent.S for agent in self.agents) / Group.n
        [agent.receive_mark(mark) for agent in self.agents]
        return mark

    def isfull(self):
        return len(self.agents) == Group.n

    def add_highest_link(self):
        # Pick highest non-allocated, positive link
        if random() > Group.gamma:
            links = sorted(self.links, key=lambda link: self.links[link], reverse=True)
            for link in links:
                new = self.graph.node[link]
                if not new.allocated and self.links[link] > 0:
                    self.add_agent(new)
                    return

        # If none, pick randomly
        ind = list(range(len(self.graph)))
        shuffle(ind)
        for i in ind:
            new = self.graph.node[i]
            if not new.allocated:
                self.add_agent(new)
                return

    def rewire(self):
        mark = self.agents[0].payoff + Agent.a * self.agents[0].S
        delta_w = Group.beta * (mark - Agent.H / 2)
        for (a1, a2) in combinations(self.agents, 2):
            n1, n2 = (a1.node, a2.node)
            if n2 in self.graph[n1]:
                self.graph[n1][n2]["weight"] += delta_w
            else:
                self.graph.add_edge(n1, n2, weight=delta_w)


class Pop:
    h0 = None

    def __init__(self, n_agents):
        pop = [Agent(1) if n < int(n_agents * Pop.h0) else Agent(0) for n in range(n_agents)]
        self.agents = pop

    def get_h(self):
        h = 0
        for agent in self.agents:
            if agent.S == Agent.H:
                h += 1
        return h / len(self.agents)


class Pop_Graph(Pop):
    alpha = None

    def __init__(self, n_agents, parameters):
        self.copying = parameters["copying"]
        super().__init__(n_agents)
        shuffle(self.agents)

        graph = nx.Graph(nx.connected_watts_strogatz_graph(n_agents, k=parameters["k"], p=parameters["p"]))
        for i, agent in enumerate(self.agents):
            graph.node[i] = agent
            agent.node = i
            agent.links = graph[i]

        for n1, n2 in graph.edges():
            graph[n1][n2]['weight'] = normalvariate(0.5, 0.1)  # random()#normalvariate(0, 1)
            graph[n1][n2]['weight'] = normalvariate(0.5, 0.1)

        self.graph = graph
        self.update_sum_links()

    def statistics(self):
        print("**Graph created**")
        print("Degree distribution:")
        print(pd.Series(list(nx.degree(self.graph).values())).value_counts())
        print("Clustering coefficient: " + str(np.array(list(nx.clustering(self.graph).values())).mean()))
        print("Average short path length: " + str(nx.average_shortest_path_length(self.graph)))
        print("Diameter: " + str(nx.diameter(self.graph)))
        print("")

    def get_h(self):
        nh = 0
        for agent in self.graph.node.values():
            if agent.S == Agent.H:
                nh += 1
        return nh / len(self.graph.node)

    def update_sum_links(self):
        for agent_i in self.graph.node:
            self.graph.node[agent_i].sum_links = sum([el["weight"] for el in self.graph[agent_i].values()])

        self.agents = sorted(self.agents, key=lambda agent: agent.sum_links, reverse=True)

    def pick_highest_node(self):
        return max(self.graph.node.values(), key=lambda agent: agent.sum_links if not agent.allocated else 0)

    def pick_highest_link(self):
        return max(self.graph.edges_iter(), key=lambda edge: self.graph[edge[0]][edge[1]]["weight"])

    def copy_strategies(self):
        def copy_by_friendship():
            for agent in self.agents:
                links_raw = self.graph[agent.node]
                weights = np.array([el["weight"] for el in links_raw.values()])

                link = choices(population=list(links_raw.keys()), weights=weights - min(weights))[0]

                agent.copy_payoff = self.graph.node[link].payoff
                agent.copy_S = self.graph.node[link].S

        def random_copying():
            for agent in self.agents:
                reference = choice(self.agents)
                agent.copy_payoff = reference.payoff
                agent.copy_S = reference.S

        if self.copying == "friendship":
            copy_by_friendship()
        elif self.copying == "random":
            random_copying()
        else:
            raise Exception("Copying behaviour not specified")

        copyh = 0
        copyl = 0
        for agent in self.agents:
            agent.allocated = False

            if Pop_Graph.alpha * (agent.copy_payoff - agent.payoff) > random():
                if agent.S == Agent.H and agent.copy_S == Agent.L:
                    copyl += 1
                elif agent.S == Agent.L and agent.copy_S == Agent.H:
                    copyh += 1

                agent.S = agent.copy_S

        return copyh, copyl

    def draw_image(self, folder_name, i):
        node_colors = [agent.S for agent in self.graph.node.values()]
        edge_colors = [self.graph[e1][e2]["weight"] for e1, e2 in self.graph.edges_iter()]
        m1, m2 = self.pick_highest_link()
        max_weight = self.graph[m1][m2]["weight"]

        plt.clf()
        nx.draw_spring(self.graph, node_color=node_colors, edge_color=edge_colors, edge_cmap=plt.cm.seismic,
                       edge_vmin=-max_weight, edge_vmax=max_weight, alpha=0.5)
        plt.savefig(folder_name + "/t" + str(i) + ".png")
