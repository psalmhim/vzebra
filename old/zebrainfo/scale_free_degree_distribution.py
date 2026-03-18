import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

def fun_coherence_level(data,dd,pp):

    ttime=len(data[0,:]);
    n1=len(data[:,0]);
   
    d1=[];
    for i in range(0,ttime-dd,pp):
        mm=np.mean(data[0:n1,i:i+dd],axis=0);
        ssum=0;

        for j in range(n1):
            c1=np.corrcoef(mm,data[j,i:i+dd])
            ssum=ssum+c1[0,1];

        d1.append(ssum/n1);

    return d1; 
def fun_draw_networkt(edges):

    G=nx.Graph()

    for i in range(len(edges)):
          G.add_edge(edges[i][0],edges[i][1]);


    nx.draw(G,node_size=3);

   
    plt.show();

    
def fun_draw_network(edges,node_pos, node_ccolor,edge_ccolor,edge_list_intra, edge_list_inter):

    G=nx.Graph()

    for i in range(len(edges)):
          G.add_edge(edges[i][0],edges[i][1]);



    pos=node_pos;#nx.spring_layout(G);  
   # nx.draw(G,pos,node_size=4,node_color='black');


    node_tt1=[];
    node_tt2=[];
    node_tt3=[];

    for i in range(300):
           node_tt1.append(i);
           node_tt2.append(i+300);
           node_tt3.append(i+600);
    

    xx=G.edges()
    pp=[(0,600)];
    
    nx.draw_networkx_edges(G,pos,edgelist=edge_list_intra,edge_color=edge_ccolor,width=2);
    nx.draw_networkx_edges(G,pos,edgelist=edge_list_inter,edge_color=[0.8,0.1,0.0,0.5],width=1);
    
    nx.draw_networkx_nodes(G, pos,node_size=1,node_color=node_ccolor);
    #nx.draw_networkx_nodes(G, pos,nodelist=node_tt3,node_size=25,node_color='Blue');

    plt.show(); 





def fun_make_small_edges(n1,p):
    
    A_net= nx.watts_strogatz_graph(n1,5,p);
    edges_t=list(A_net.edges);
    edges=[];


    for i in range(len(edges_t)):
        if(edges_t[i][0]!=edges_t[i][1]):
      
           check=0; 
           for j in range(len(edges)):

                if(edges[j][0]==edges_t[i][0] and edges[j][1]==edges_t[i][1]):
                  #  print("2",edges[j][1], edges_t[i][1])
                    check=1;

                if(edges[j][1]==edges_t[i][0] and edges[j][0]==edges_t[i][1]):
                  #  print("2",edges[j][1], edges_t[i][1])
                    check=1;
           if(check==0):

               if(random.random()<0.5):
                   edges.append([int(edges_t[i][0]),int(edges_t[i][1])]);
               else:
                   edges.append([int(edges_t[i][1]),int(edges_t[i][0])]);
                   

    return edges;



def fun_make_random_edges(n1):
    
    A_net= nx.erdos_renyi_graph(n1,0.080);
    edges_t=list(A_net.edges);
    edges=[];


    for i in range(len(edges_t)):
        if(edges_t[i][0]!=edges_t[i][1]):
      
           check=0; 
           for j in range(len(edges)):

                if(edges[j][0]==edges_t[i][0] and edges[j][1]==edges_t[i][1]):
                  #  print("2",edges[j][1], edges_t[i][1])
                    check=1;

           if(check==0):
               edges.append(edges_t[i]) 

    return edges;


def fun_make_scale_edges(n1):
    
    A_net= nx.scale_free_graph(n1);
    edges_t=list(A_net.edges);
    edges=[];


    for i in range(len(edges_t)):
        if(edges_t[i][0]!=edges_t[i][1]):
      
           check=0; 
           for j in range(len(edges)):

                if(edges[j][0]==edges_t[i][0] and edges[j][1]==edges_t[i][1]):
                  #  print("2",edges[j][1], edges_t[i][1])
                    check=1;
                
                if(edges[j][1]==edges_t[i][0] and edges[j][0]==edges_t[i][1]):
                  #  print("2",edges[j][1], edges_t[i][1])
                    check=1;

                    
           if(check==0):

               if(random.random()<0.5):
                   edges.append([edges_t[i][0],edges_t[i][1]]);
               else:
                   edges.append([edges_t[i][1],edges_t[i][0]]);
                   
               #edges.append(edges_t[i]) 
   
        

    return edges


def fun_make_edges(n1):
    
    A_net= nx.scale_free_graph(n1);
    edges_t=list(A_net.edges);
    edges=[];


    for i in range(len(edges_t)):
        if(edges_t[i][0]!=edges_t[i][1]):
      
           check=0; 
           for j in range(len(edges)):

                if(edges[j][0]==edges_t[i][0] and edges[j][1]==edges_t[i][1]):
                  #  print("2",edges[j][1], edges_t[i][1])
                    check=1;

           if(check==0):
               edges.append(edges_t[i]) 

    return edges;


def fun_chose_edges1(edges, n1, sn,pp):

    edges_degree=np.zeros(n1);

    for j in range(len(edges)):
        xx=edges[j][0];
        yy=edges[j][1];
        edges_degree[xx]+=1;
        edges_degree[yy]+=1;
        
    xx=edges_degree.argsort();
    yy=np.sort(edges_degree);

    xx=xx[::-1]
    yy=yy[::-1]

    return (xx[2*sn:(2*(sn)+pp)]), (yy[2*sn:(2*(sn)+pp)])

def fun_chose_edges(edges, n1, sn,pp):

    edges_degree=np.zeros(n1);

    for j in range(len(edges)):
        xx=edges[j][0];
        yy=edges[j][1];
        edges_degree[xx]+=1;
        edges_degree[yy]+=1;
        
    xx=edges_degree.argsort();
    yy=np.sort(edges_degree);

    xx=xx[::-1]
    yy=yy[::-1]

    return (xx[20*sn:(20*(sn)+pp)], yy[20*sn:(20*(sn)+pp)])

def fun_sym_construct_total_edges(area_num,zebra_edges,sc_mat,start_pos, target_pos,n1,nc):

  
    #    zebra_edges=[];
    #    for i in range(area_num):
    #        edges=fun_make_edges(n1);
        #zebra_edges.append(edges);
    #for i in range(area_num):
    #   sn=0;
      # xx,yy=fun_chose_edges(zebra_edges[i], n1, sn);    
        
    total_edges=[];

    for i in range(area_num):
        for j in range(len(zebra_edges[i])):

           total_edges.append([zebra_edges[i][j][1]+i*n1,zebra_edges[i][j][0]+i*n1]);


    scc=20;
    [xic,ic]=fun_chose_edges1(zebra_edges[0], n1, start_pos,scc);
    [xjp,jp]=fun_chose_edges1(zebra_edges[0], n1, target_pos,scc);

    lk1=[];
    lk2=[];
    
    for i in range(nc):
        k1=random.randint(0,scc-1);
        k2=random.randint(0,scc-1);
        lk1.append(k1);
        lk2.append(k2);
            
    

    for i in range(area_num):
         for j in range(area_num):

             if(sc_mat[i][j]>0):           


                scc=20;
           

                if(target_pos>-1):
                    a=1;
                    #[xic,ic]=fun_chose_edges1(zebra_edges[i], n1, start_pos,scc);   
                    #[xip,ip]=fun_chose_edges1(zebra_edges[i], n1, target_pos,scc);

    
                    #[xjc,jc]=fun_chose_edges1(zebra_edges[j], n1, start_pos,scc);   
                    #[xjp,jp]=fun_chose_edges1(zebra_edges[j], n1, target_pos,scc);
                  #  print(i,j);
                  # print("1 : ",xic);
                  #  print("2 : ",xip);
                  #  print("3 : ",xjc);
                  #  print("4 : ",xjp);

                if(target_pos==-1):

                    xic=[];
                    xip=[];
                    xjc=[];
                    xjp=[];

                    for kk in range(n1):
                       xic.append(kk);
                       xip.append(kk);
                       xjc.append(kk);
                       xjp.append(kk);
                      
                   


                #random.shuffle(xic);
                #random.shuffle(xip);
                #random.shuffle(xjc);
                #random.shuffle(xjp);
                

                for k in range(nc):
                    pp=1;
                    #k1=random.randint(0,scc-1);
                    #k2=random.randint(0,scc-1);

                                       
                    #if(random.random()<0.5):
                       #total_edges.append([i*n1+xic[lk1[k]],j*n1+xjp[lk2[k]]]);
                      # total_edges.append([j*n1+xjp[lk2[k]],i*n1+xic[lk1[k]]]);
                    #e
                    #else:
                     #  total_edges.append([j*n1+xjc[k1],i*n1+xip[k2]]);
                   
    

    r_total_edges=[];

    nn=area_num*n1;
    t_mat=np.zeros((nn,nn));

    for j in total_edges:

        t_mat[j[0],j[1]]=1;
        t_mat[j[1],j[0]]=1;

    for i in range(nn):
        for j in range(i+1,nn):

            if(t_mat[i,j]>0):
                if(random.random()<0.5):
                    r_total_edges.append([i,j]);
                else:
                    r_total_edges.append([j,i]);
        
                                
                    
    return total_edges



def fun_zebrafish_construct_total_edges(area_num,zebra_edges,sc_mat,start_pos, target_pos,n1,nc):

  
#    zebra_edges=[];
    
#    for i in range(area_num):
#        edges=fun_make_edges(n1);
        #zebra_edges.append(edges);

    for i in range(area_num):
        sn=0;
      # xx,yy=fun_chose_edges(zebra_edges[i], n1, sn);    
        
    total_edges=[];

    for i in range(area_num):
        for j in range(len(zebra_edges[i])):

          #  if(random.random()<0.5):  
                 total_edges.append([zebra_edges[i][j][0]+i*n1,zebra_edges[i][j][1]+i*n1]);
          #  else:
          #     total_edges.append([zebra_edges[i][j][1]+i*n1,zebra_edges[i][j][0]+i*n1]);


    for i in range(area_num):
         for j in range(area_num):

             if(sc_mat[i][j]>0):           


             
          
                [xic,ic]=fun_chose_edges1(zebra_edges[i], n1, start_pos,nc);   
                [xjp,jp]=fun_chose_edges1(zebra_edges[j], n1, target_pos,nc);

    
                #[xjc,jc]=fun_chose_edges(zebra_edges[j], n1, start_pos,scc);   
                #[xjp,jp]=fun_chose_edges(zebra_edges[j], n1, target_pos,scc);

                #random.shuffle(xic);
                #random.shuffle(xip);
                #random.shuffle(xjc);
                #random.shuffle(xjp);

                tscc=int(nc*sc_mat[i][j]/283);
                #print(tscc);
                for k in range(tscc):

                    k1=random.randint(0,nc-1);
                    k2=random.randint(0,nc-1);
                 
                                       
                   # if(random.random()<0.5):
                    total_edges.append([i*n1+xic[k1],j*n1+xjp[k2]]);

                    #else:
                     #  total_edges.append([j*n1+xjc[k1],i*n1+xip[k2]]);
                   

    #r_total_edges=[];

    #nn=area_num*n1;
    #t_mat=np.zeros((nn,nn));

    #for j in total_edges:

       # t_mat[j[0],j[1]]=1;
       # t_mat[j[1],j[0]]=1;

                                  
                    
    return total_edges;

def fun_construct_random_total_edges(start_pos, target_pos):
    
    n1=300;

    area_num=72;

    zebra_edges=[];
    

    for i in range(area_num):
        edges=fun_make_random_edges(n1);
        zebra_edges.append(edges);


    for i in range(area_num):
        sn=0;
       # xx,yy=fun_chose_edges(zebra_edges[i], n1, sn);    
        #print(xx);
        #print(yy);

    sc_mat=np.load('./data/zebrafish_sc.npy');


    total_edges=[];

    for i in range(area_num):
        for j in range(len(zebra_edges[i])):

            if(random.random()<0.5):  
                total_edges.append([zebra_edges[i][j][0]+i*n1,zebra_edges[i][j][1]+i*n1]);
            else:
                total_edges.append([zebra_edges[i][j][1]+i*n1,zebra_edges[i][j][0]+i*n1]);


    for i in range(area_num):
         for j in range(area_num):

             if(sc_mat[i][j]>0):           


                scc=(sc_mat[i][j]/285)*30;
                
                
                [xic,y1]=fun_chose_edges(zebra_edges[i], n1, start_pos,30);   
                [xip,y1]=fun_chose_edges(zebra_edges[i], n1, target_pos,30);


                [xjc,y1]=fun_chose_edges(zebra_edges[j], n1, start_pos,30);   
                [xjp,y1]=fun_chose_edges(zebra_edges[j], n1, target_pos,30);


                random.shuffle(xic);
                random.shuffle(xip);
                random.shuffle(xjc);
                random.shuffle(xjp);
                

                for k in range(int(scc)):
                    
                    if(random.random()<0.5):
                       total_edges.append([i*n1+xic[k],j*n1+xjp[k]]);
                    else:
                       total_edges.append([j*n1+xjc[k],i*n1+xip[k]]);
                   


                                
                    
    return total_edges



             

def fun_construct_total_edges(start_pos, target_pos):
    
    n1=300;

    area_num=72;

    zebra_edges=[];

    for i in range(area_num):
        edges=fun_make_edges(n1);
        zebra_edges.append(edges);


    for i in range(area_num):
        sn=0;
       # xx,yy=fun_chose_edges(zebra_edges[i], n1, sn);    
        #print(xx);
        #print(yy);

    sc_mat=np.load('./data/zebrafish_sc.npy');


    total_edges=[];

    for i in range(area_num):
        for j in range(len(zebra_edges[i])):

            if(random.random()<0.5):  
                total_edges.append([zebra_edges[i][j][0]+i*n1,zebra_edges[i][j][1]+i*n1]);
            else:
                total_edges.append([zebra_edges[i][j][1]+i*n1,zebra_edges[i][j][0]+i*n1]);


    for i in range(area_num):
         for j in range(area_num):

             if(sc_mat[i][j]>0):           


                scc=(sc_mat[i][j]/285)*30;
                
                
                [xic,y1]=fun_chose_edges(zebra_edges[i], n1, start_pos,30);   
                [xip,y1]=fun_chose_edges(zebra_edges[i], n1, target_pos,30);


                [xjc,y1]=fun_chose_edges(zebra_edges[j], n1, start_pos,30);   
                [xjp,y1]=fun_chose_edges(zebra_edges[j], n1, target_pos,30);


                random.shuffle(xic);
                random.shuffle(xip);
                random.shuffle(xjc);
                random.shuffle(xjp);
                

                for k in range(int(scc)):
                    
                    if(random.random()<0.5):
                       total_edges.append([i*n1+xic[k],j*n1+xjp[k]]);
                    else:
                       total_edges.append([j*n1+xjc[k],i*n1+xip[k]]);
                   


                                
                    
    return total_edges
    

def fun_for_link_threes_module(edges,start_pos,end_pos,nc,n1):
    
    area_num=3;
    #n1=300;

    zebra_edges=[];
    
    #edges=scc.fun_make_scale_edges(n1);

    zebra_edges.append(edges);
    zebra_edges.append(edges);
    zebra_edges.append(edges);

    sc_mat=[[0,1,1],[1,0,1],[1,1,0]];
    
    edges=fun_sym_construct_total_edges(area_num,zebra_edges,sc_mat,start_pos, end_pos,n1,nc);

    return edges;






   

#edges=fun_construct_total_edges();

    


 

