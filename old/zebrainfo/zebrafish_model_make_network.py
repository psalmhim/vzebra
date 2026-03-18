import networkx as nx
import networkx.algorithms as na
import matplotlib.pyplot as plt 
import numpy as np
import random
import copy
import scale_free_degree_distribution as scc

def fun_make_scale_zebra_model(sc_mat,nn,sel):

    n1=300;
    nc=10;
    degree_classify=[];
    total_edges=[];
    
    for ii in range(72):
        
        t_scale_edges=scc.fun_make_scale_edges(n1);
        #scale1_edges=fun_for_scale_run(t_scale_edges,0,60,nc);

        for ee in t_scale_edges:
            total_edges.append([ee[0]+ii*300,ee[1]+ii*300]);
            
        degree_num=np.zeros([300,1]);

        for ee in t_scale_edges:
           degree_num[ee[0]]+=1;
           degree_num[ee[1]]+=1;

        core_peri=[];
        for ee in degree_num:
            if(ee==0):
                core_peri.append(0);
            else:
                core_peri.append(1/(ee));
        
        high_degree=[];
        middle_degree=[];
        low_degree=[];

        for i in range(300):

            if(core_peri[i]<=0.2):
                high_degree.append(i);
            if(core_peri[i]>0.2 and core_peri[i]<=0.8):    
                middle_degree.append(i);
            if(core_peri[i]>0.8):    
                low_degree.append(i);
        a=[];
        a.append(high_degree);
        a.append(middle_degree);
        a.append(low_degree);
        degree_classify.append(a);
    
    nc=10;
    in_scale1_edges=[];

    #sc_mat=[[0,1,1],[1,0,1],[1,1,0]];
    s_max=np.log10(np.max(sc_mat));
    for i in range(72):
        for j in range(72):
           if(sc_mat[i][j]>0):
                nc=int(13/s_max*np.log10(sc_mat[i][j])+2);

     
                
                
                np.random.shuffle(degree_classify[i][0]);
                #np.random.shuffle(degree_classify[i][1]);
                #np.random.shuffle(degree_classify[i][2]);

                np.random.shuffle(degree_classify[j][0]);
                np.random.shuffle(degree_classify[j][1]);
                np.random.shuffle(degree_classify[j][2]);

                for k in range(nc):
                    if(sel==0):
                        total_edges.append([degree_classify[i][0][k]+i*300,degree_classify[j][0][k]+j*300]);
                    if(sel==1):
                        total_edges.append([degree_classify[i][0][k]+i*300,degree_classify[j][1][k]+j*300]);
                    if(sel==2):
                        total_edges.append([degree_classify[i][0][k]+i*300,degree_classify[j][2][k]+j*300]);

    return total_edges;


i=1;#0:core-core, 1:core-meidum peri, 2:core-low peri,
sc_mat=np.loadtxt('./zebra_sc.txt');
total_edges=fun_make_scale_zebra_model(sc_mat,72,i);

sc_mat=np.zeros([72*300,72*300]);

for i in range(len(total_edges)):
    xx=total_edges[i][0];
    yy=total_edges[i][1];

    sc_mat[xx][yy]=1;

plt.imshow(sc_mat);
plt.show();












