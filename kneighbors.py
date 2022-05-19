import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from numba import jit
import seaborn as sns
import pandas as pd
import random

@jit
def make_d(d,A,N):#次数行列を作成
    for i in range(N):
        for j in range(N):
            if A[i,j]==1:
                d[i,i]=d[i,i]+1

@jit
def min_d(X,d):#最大のノード次数を取得
    (size1,size2)=X.shape
    min=1
    max=0
    index=0
    for i in range(size1):
        if d[i][i]>max:
            max=d[i][i]
            index=i
    print("index",index)
    print("max",d[index][index])
    return index       

@jit
def adj(N,A,u,index):#隣接行列に初期値を与える
    #for i in range(N):
        #if A[index,i]==1 or i==index:
            #u[i]=0.1
    u[index]=0.3

@jit
def laplacian(s,L):#ラプラシアンを求める
    L1=s.shape
    S=int(s.size)
    ts = np.zeros(S)
    for i in range(S):
        for j in range(S):
            ts[i]+=L[i,j]*s[j]
    return ts

@jit
def calc(a, h, a2, h2, La,c):#状態量を求める
    L = a.size
    (L2,L2)=La.shape
    dt=0.1
    Dh=0.5#パラメーター始
    ca=0.08
    ch=0.11
    da=0.08
    #dh=0
    mua=0.03
    muh=0.12
    #aとhの密度が0.1になるように設定
    #roa=0.003
    #roh=0.001
    roa=(da+mua-ca)/10
    roh=(muh-ch)/10
    #roa=mua/10
    #roh=muh/10
    fa=ca-mua
    fh=-da
    ga=ch
    gh=-muh
    if c==0:
        Da=0.059
    elif c==1:
        Da=(Dh*(fa*gh-2*fh*ga)-2*Dh*np.sqrt(fh*ga*fh*ga-fh*ga*fa*gh))/(gh*gh)
    elif c==2:
        Da=0.057
    elif c==3:
        Da=0.02
    mina=0
    minh=0
    maxa=1
    maxh=1 
    sa = (ca*a)-(da*h)+roa-mua*a -Da * laplacian(a,La) ##反応項と拡散項を計算
    sh = (ch*a)+roh-muh*h -Dh * laplacian(h,La)  
    for i in range(L):
            a2[i] = a[i]+(sa[i])*dt #-mua*a[i,j]
            h2[i] = h[i]+(sh[i])*dt # -muh*h[i,j]           
            if a2[i]<mina:
                a2[i]=mina
            if h2[i]<minh:
                h2[i]=minh
            if a2[i]>maxa:
                a2[i]=maxa
            if h2[i]>maxh:
                h2[i]=maxh


def pic(N,u,v,G,pos,indexlist):#図示する
    for j in range(N):
        u[j]=round(u[j],2)
        v[j]=round(v[j],2)      
    print("maxu",np.max(u),"minu",np.min(u),"maxv",np.max(v),"minv",np.min(v))
    cent=u
    node_size = list(map(lambda x:x*500, cent))
    nodes = nx.draw_networkx_nodes(G, pos,node_size=30,
                               cmap='cool',
                               node_color=list(cent),
                               nodelist=list(indexlist))
    edges = nx.draw_networkx_edges(G, pos, width = 0.25)
    plt.colorbar(nodes)
    plt.show()
    cent1=v
    node_size = list(map(lambda x:x*500, cent))
    nodes = nx.draw_networkx_nodes(G, pos,node_size=30,
                               cmap='cool',
                               node_color=list(cent1),
                               nodelist=list(indexlist))
    edges = nx.draw_networkx_edges(G, pos, width = 0.25)
    plt.colorbar(nodes)
    plt.show()  



def main():
    N = 1000# the number of points
    indexlist=np.zeros(N)
    for i in range(N):
      indexlist[i]=i
    np.random.seed(seed=0)  
    X = np.random.rand(N, 2) # N行2列の配列を作成
    A = kneighbors_graph(X, n_neighbors=4, mode='connectivity').toarray() # (asymmetric) adjacecy matrix#mode=connectivity(０と1の接続行列)
    #A=nx.random_regular_graph(4, N, seed=None)
    edges = []                                                              
    d=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if A[i, j] != 0:
               edges.append((i, j)) #どのノードとどのノードがつながっているかのリスト作成

    ### Creating k-nearest neighbor graph from edge lists
    G = nx.Graph()
    #G = nx.random_regular_graph(4, N, seed=None)
    for i in range(N):
        G.add_node(i, pos=X[i]) # setting (x, y)-coordinates of nodes
    G.add_edges_from(edges)
    pos = {
        i: (X[i][0], X[i][1])
        for i in range(N)
    }
    ary = nx.to_numpy_matrix(G)
    make_d(d,ary,N)
    L=d-ary
    index=min_d(X,d)#最大次数を求める
    u0= np.zeros(N)#最大次数のノードと隣接してる行列の値に初期値を与える
    a=adj(N,ary,u0,index)
    u02 =np.zeros(N) 
    v0 = np.zeros(N)+0.1
    v02 =np.zeros(N)
    plt.subplot()
    plt.figure(figsize=(6,4))
    nx.draw(G, nx.get_node_attributes(G, 'pos'), node_size=20)
    plt.tight_layout()
    plt.show()
    time=100000
    for k in range(4):
        a=np.zeros(N)
        a2=np.zeros(N)
        h=np.zeros(N)
        h2=np.zeros(N)
        for i in range(N):#これができないと配列が初期化できない
            a[i]=u0[i]
            a2[i]=u02[i]
            h[i]=v0[i]
            h2[i]=v02[i]
        for i in range(time):
            if i % 2 == 0:
                calc(a, h, a2, h2, L,k)
            else:
                calc(a2, h2, a, h, L,k)
                    #現在のステップの状態u2,v2から次のステップの状態u,vを計算する
            if i==0 and k==0:   
                    pic(N,a,h,G,pos,indexlist)
            if i==time-1:   
                    pic(N,a,h,G,pos,indexlist)
                        
main=main()
