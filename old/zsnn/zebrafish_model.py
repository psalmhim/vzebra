import networkx as nx
import networkx.algorithms as na
import matplotlib.pyplot as plt 
import numpy as np
import random
import copy
import MONET_SNN_CUDA_PYTHON as snn
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




def fun_cdua_run(edges,coupling_const):
    
    area_num=72;
    n1=300;

    _neurnal=snn.DM_SNN();
    _neurnal.all_clear();

    nn=n1*area_num;
    _neurnal.set_neuron_number(nn);

    for i in range(nn):
        if random.random()<0.2:
          _neurnal.set_inhibtion_neuron(i);
       

    
    for j in range(len(edges)):

        xx=int(edges[j][0]);
        _neurnal.set_connection(int(edges[j][0]),int(edges[j][1]),coupling_const);
        #0.5     

    time=10;

    ll=20*1000*time;


   
    for i in range(nn):
      _neurnal.set_noise(i, 10.6);


              
    _neurnal.set_run_param(0.05,0,ll,True,False,False);

    _neurnal.create_cuda_memory();
    _neurnal.set_calcium_recording(100);

    _neurnal.cuda_run_python();


    spike_data=_neurnal.get_spike_data();   
    ca_data=_neurnal.get_ca_data();
    ca_data=np.array(ca_data);

    spike_data=np.array(spike_data);


    return spike_data,ca_data

def  fun_analysis_1(load_file_name,save_file_name):
    
    ca_data=np.load(load_file_name);
    ca_data=np.transpose(ca_data);
    ca_data=ca_data+np.random.rand(300*72,len(ca_data[0]));

    mean_ca_data=[];
    for i in range(72):
        xx=np.mean(ca_data[i*300:(i+1)*300,:],axis=0);
        mean_ca_data.append(xx);

    mean_ca_data=np.array(mean_ca_data);
    for i in range(72):
        plt.plot(mean_ca_data[i]-0.5*i);
    plt.show();

    t_time=len(mean_ca_data[i]);

    window=70;
    dd=60;
    corr_data=[];
    for i in range(0,t_time-window,dd):
        cc1=np.corrcoef(mean_ca_data[:,i:i+window]);

        tt=[];
        for j in range(72):
            for k in range(j+1,72):
                tt.append(cc1[j,k]);
        corr_data.append(tt);        


    fcc=np.corrcoef(corr_data);

    for j in range(len(fcc)):
        fcc[j,j]=0.0;

    np.savetxt(save_file_name,fcc,delimiter=' ');

            
    plt.imshow(fcc);
    plt.show();


def fun_figure_one_scale_model():

    n1=300;
    nc=10;
    t_scale_edges=scc.fun_make_scale_edges(n1);
    scale1_edges=fun_for_scale_run(t_scale_edges,0,60,nc);

    degree_num=np.zeros([300,1]);

    for ee in t_scale_edges:
       degree_num[ee[0]]+=1;
       degree_num[ee[1]]+=1;

    core_peri=[];
    for ee in degree_num:
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

    np.random.shuffle(high_degree);
    np.random.shuffle(middle_degree);
    np.random.shuffle(low_degree);

    nc=10;
    in_scale1_edges=[];

    sc_mat=[[0,1,1],[1,0,1],[1,1,0]];

    for i in range(3):
        for j in range(3):
           if(sc_mat[i][j]>0):
             
                for k in range(nc):
                    if(sel==0):
                        in_scale1_edges.append([high_degree[k]+i*300,high_degree[k]+j*300]);
                    if(sel==1):
                        in_scale1_edges.append([high_degree[k]+i*300,middle_degree[k]+j*300]);
                    if(sel==2):
                        in_scale1_edges.append([high_degree[k]+i*300,low_degree[k]+j*300]);

    return scale1_edges, in_scale1_edges,degree_num;


#fun_analysis_1('./zebra_scae_2.npy','./zebra_fcc_2.txt');
#aaaa

sc_mat=np.load('./zebra_sc.npy');
#plt.imshow(sc_mat);
#plt.show();


for i in range(1):
    #i =0; core-core
    #i=1; core-medium 
    #i=2:core-preiphral 
    for j in range(1):

        total_edges=fun_make_scale_zebra_model(sc_mat,72,i);
        g=1.38;
        spike_data,ca_data=fun_cdua_run(total_edges,g);

        file_name1='./zebrafish_scale_large_scale_ca_data_'+str(i)+'_iter_'+str(j)+'.npy'
        print(file_name1)
        np.save(file_name1,ca_data);




plt.scatter(spike_data[:,1],spike_data[:,0],s=1);
plt.show();


aaaaaaaaaa;












