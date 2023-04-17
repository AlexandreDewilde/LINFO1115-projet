import pandas as pd
import numpy as np
from template_utils import *
import sys 
import networkx as nx
from collections import deque
sys.setrecursionlimit(6000)

def dfs(x, vis, g):
    vis[x] = True
    for adj in g[x]:
        if vis[adj]:
            continue
        dfs(adj, vis, g)
        
def dfs_bridge(x, p, t, vis, tin, low, g, b):
    vis[x] = True
    tin[x] = low[x] = t[0]
    t[0] += 1
    for adj in g[x]:
        if adj == p:
            continue
        if vis[adj]:
            low[x] = min(low[x], tin[adj])
        else:
            dfs_bridge(adj, x, t, vis, tin, low, g, b)
            low[x] = min(low[x], low[adj])
            if low[adj] > tin[x]:
                b.append((x, adj))
def bfs(x, g, ignore=-1):
    q = deque([x])
    dst = [-1]*len(g)
    dst[x] = 0
    while q:
        x = q.popleft()
        for adj in g[x]:
            if dst[adj] == -1 and (x,adj) != ignore and (adj, x) != ignore:
                dst[adj] = dst[x] + 1
                q.append(adj)
    return dst

def Q1(dataframe):
    g = {}
    for row in dataframe.iterrows():
        if (row[1]["Src"]) not in g:
            g[(row[1]["Src"])] = set()
        if (row[1]["Dst"]) not in g:
            g[(row[1]["Dst"])] = set()
        g[(row[1]["Src"])].add(row[1]["Dst"])
        g[(row[1]["Dst"])].add(row[1]["Src"])
        
    visited = [False]*len(g)
    tin = [-1]*len(g)
    low = [-1]*len(g)
    bridges = []
    connected = 0
    comps = []
    for i in range(len(g)):
        if not visited[i]:
            connected += 1
            comps.append(set())
            dfs_bridge(i, -1, [0], visited, tin, low, g, bridges)
    local = 0
    for i in range(len(g)):
        for adj in g[i]:
            if adj < i:
                common = g[i].intersection(g[adj])
                local += len(common) == 0
    return [connected, len(bridges), local]

def Q2(dataframe):
    df = dataframe[dataframe["Time"] >= dataframe["Time"].median()].drop_duplicates(["Src", "Dst"], keep="first")
    g = {}
    timing = {}
    for _, (src, dst, time) in df.iterrows():
        if src not in g:
            g[src] = set()
        if dst not in g:
            g[dst] = set()
        g[src].add(dst)
        g[dst].add(src)
        if (src, dst) not in timing and (dst, src) not in timing:
            timing[(src, dst)] = time
    triadic = 0
    for i in range(len(g)):
        if i not in g:
            continue
        for adj in g[i]:
            for adj2 in g[i]:
                if adj != adj2 and adj2 not in g[adj]:
                    triadic += 1
    return triadic #total number of triadic closures created after the median timestamp

def Q3(dataframe):
    #Your code here
    return [32, 19, 32, 29, 12] #at index 0 the number of shortest paths of lenght 0, at index 1 the number of shortest paths of length 1, ...

def Q4(dataframe):
    #Your code here
    return [10, 0.2413] # the id of the node with the highest pagerank score, the associated pagerank value.
    #Note that we consider that we reached convergence when the sum of the updates on all nodes after one iteration of PageRank is smaller than 10^(-10)

def Q5(dataframe):
    #Your code here
    return 0.5555 #the average local clustering coefficient of the graph

#you can write additionnal functions that can be used in Q1-Q5 functions in the file "template_utils.py", a specific place is available to copy them at the end of the Inginious task.

df = pd.read_csv('CollegeMsg.csv')
print(Q1(df))
print(Q2(df))
print(Q3(df))
print(Q4(df))
print(Q5(df))
