import pandas as pd
import numpy as np
from template_utils import *
import sys 
import networkx as nx
from collections import deque
sys.setrecursionlimit(6000)

def dfs(x, vis, g, comp):
    vis.add(x)
    comp.append(x)
    for adj in g[x]:
        if adj in vis:
            continue
        dfs(adj, vis, g, comp)
        
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

def bfs_dst(x, g):
    q = deque([x])
    dst = {}
    dst[x] = 0
    while q:
        x = q.popleft()
        for adj in g[x]:
            if adj not in dst:
                dst[adj] = dst[x] + 1
                q.append(adj)
    return dst

def Q1(dataframe):
    g = {}
    for _, (src, dst, _) in df.iterrows():
        if src not in g:
            g[src] = set()
        if dst not in g:
            g[dst] = set()
        g[src].add(dst)
        g[dst].add(src)
        
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
    df = dataframe[dataframe["Time"] >= dataframe["Time"].median()].drop_duplicates(["Src", "Dst"], keep="first").sort_values("Time")
    timing = {}
    g = {}
    for _, (src, dst, time) in df.iterrows():
        if (src, dst) not in timing and (dst, src) not in timing:
            timing[(src, dst)] = time
        # g[src].add(dst)
        # g[dst].add(src)
    triadic = 0
    for (src, dst), _ in timing.items():
        for u in g.get(src, []):
            for v in g.get(dst, []):
                if u == v:
                    triadic += 0.5
        if src not in g:
            g[src] = set()
        if dst not in g:
            g[dst] = set()
        g[src].add(dst)
        g[dst].add(src)
    return int(triadic)

def Q3(dataframe):
    g = {}
    g2 = {}
    for _, (src, dst, _) in df.iterrows():
        if src not in g:
            g[src] = set()
        if src not in g2:
            g2[src] = set()
        if dst not in g2:
            g2[dst] = set()
        g[src].add(dst)
        g2[src].add(dst)
        g2[dst].add(src)
    comps = []
    vis = set()
    for i in range(len(g2)):
        if i not in vis:
            comps.append([])
            dfs(i, vis, g2, comps[-1])
    comps.sort(key=lambda x: len(x))
    res = {}
    for node in comps[-1]:
        dst = bfs_dst(node, g2)
        for val in dst.values():
            res[val] = res.get(val, 0) + 1
    # print(len(g))
    return [res.get(i, 0) for i in range(max(res.keys()) + 1)]

def Q4(dataframe):
    g = {}
    g_inv = {}
    nodes = set()
    for _, (src, dst, _) in df.iterrows():
        nodes.add(src)
        nodes.add(dst)
        if src not in g:
            g[src] = set()
        if dst not in g_inv:
            g_inv[dst] = set()
        g[src].add(dst)
        g_inv[dst].add(src)
    d = 0.85
    N = len(nodes)
    pr = {i:1/N for i in nodes}
    while True:
        pr_new = {}
        for p in nodes:
            pr_new[p] = (1 - d) / N
            if p not in g_inv:
                continue
            for adj in g_inv[p]:
                pr_new[p] += d * pr[adj] / len(g[adj])
        if sum(abs(pr_new[i] - pr[i]) for i in nodes) < 1e-10:
            pr = pr_new
            break
        pr = pr_new
    idx, val = max(pr.items(), key=lambda x: x[1])
    tot = sum(map(abs,pr.values()))
    # g = nx.from_pandas_edgelist(dataframe, "Src", "Dst", create_using=nx.DiGraph())
    # print(max(nx.pagerank(g, alpha = 0.85, tol=1e-10).items(), key=lambda x:x[1]))
    return [idx, val/tot] # the id of the node with the highest pagerank score, the associated pagerank value.
    #Note that we consider that we reached convergence when the sum of the updates on all nodes after one iteration of PageRank is smaller than 10^(-10)

def Q5(dataframe):
    g = {}
    for _, (src, dst, _) in df.iterrows():
        if src not in g:
            g[src] = set()
        if dst not in g:
            g[dst] = set()
        g[src].add(dst)
        g[dst].add(src)
    triangles = {}
    for node in g:
        for u in g[node]:
            for v in g[node]:
                if u == v:
                    continue
                if u in g[v]:
                    triangles[node] = triangles.get(node, 0) + 0.5
    avg = 0
    for u in g:
        if len(g[u]) -1:
            avg += 2 * triangles.get(u,0) / len(g) / len(g[u]) / (len(g[u]) - 1)

    return avg #the average local clustering coefficient of the graph

#you can write additionnal functions that can be used in Q1-Q5 functions in the file "template_utils.py", a specific place is available to copy them at the end of the Inginious task.

df = pd.read_csv('CollegeMsg.csv')
# print(Q1(df))
print(Q2(df))
print(Q3(df))
print(Q4(df))
print(Q5(df))
