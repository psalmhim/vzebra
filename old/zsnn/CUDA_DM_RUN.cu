#include "CUDA_DM_RUN.cuh"

#include <cstdio>
//#include <boost/thread.hpp>
#include <random>
#include <iostream>
//#include <boost/atomic.hpp
#include <array>
#include <cuda_fp16.h>


void s_neuronal_netowrk::fun_set_weight(int from, int to,double new_w)
{
    int nn = this->nn_neuron_num;
    
    int i = to;

    int st = 0;
    if (i - 1 < 0) st = 0;
    else
        st = this->v_nn[i - 1];
        
    double t_sum = 0;
    int k = 0;
    for (int j = st; j < this->v_nn[i]; j++)
    {
            int ii = this->v_pred_id[j];
     
            if (from == ii)
                this->v_weight[j] = new_w;

    }
    


}

void s_neuronal_netowrk::fun_hebbian_rate_spike(int window) 
{
    int nn = this->nn_neuron_num;

    int tt = this->run_param.ft - this->run_param.st;

    int time_ll = tt / window;


    //typedef boost::multi_array<int,3 > array_type;
    //typedef array_type::index index;

    std::vector<int> spike_mat(nn * time_ll, 0);
    //array_type spike_mat(boost::extents[3][4][2]);


    for (int i = 0; i < this->out_data.size(); i++)
    {
          int tx=out_data[i].x;
          int ty=(out_data[i].y-this->run_param.st)/window;
          
          if(ty>=0)  spike_mat[tx*time_ll+ty] = 1;
    }

    for (int i = 0; i < nn; i++)
    {
        int st = 0;
        if (i - 1 < 0) st = 0;
        else
            st = this->v_nn[i - 1];


        auto fun_spike_corr = [&spike_mat,time_ll](int ii, int jj)
            {
                int temp = 0;
                for (int i = 0; i < time_ll; i++)
                {

                    temp+=spike_mat[ii * time_ll + i] * spike_mat[jj * time_ll + i];

                }
                
                return temp;
            };

        auto fun_spike_sum = [&spike_mat, time_ll](int ii)
            {
                int temp = 0;
                for (int i = 0; i < time_ll; i++)
                {

                    temp += spike_mat[ii * time_ll + i];

                }

                return temp;
            };


        for (int j = st; j < this->v_nn[i]; j++)
        {
            int ii = this->v_pred_id[j];
            int jj = i;


            double sc=(double)fun_spike_corr(ii, jj);
            double sum_ii = (double)fun_spike_sum(ii);
            double sum_jj = (double)fun_spike_sum(jj);

            if (sc > 2.0)
                this->v_weight[j] *= 1.1;
            //else
              //  this->v_weight[j] *= 0.95;
            //this->v_weight[j] += 0.005 * (sum_ii-19)* (sum_jj - 19);
            
            
            if (this->v_weight[j] < 0)    
                this->v_weight[j] = 0.0;
            //else
               // this->v_weight[j] *= 0.95;
        }
    }
    
    //xx;

    return;
}


__global__ void setup_rand(curandState* state, int seed, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
        curand_init(seed + tid, tid, 0, &state[tid]);
}


void s_neuronal_netowrk::create_cuda_memory()
{

    this->vt_spike_check.resize(this->nn_neuron_num, 0);

            
    //std::vector<int> v_nn;
    //std::vector<int> v_pred_id;
    //std::vector<double> v_weight;

    v_nn.resize(this->nn_neuron_num, 0);

    v_nn[0] = this->_connect_data[0].s_pre_id.size();

    for (int i = 1; i < this->_connect_data.size(); i++)
    {
       v_nn[i]=v_nn[i-1]+ this->_connect_data[i].s_pre_id.size();
    }

    
    for (int i = 0; i < this->_connect_data.size(); i++)
    {
        for (int j = 0; j < this->_connect_data[i].s_pre_id.size(); j++)
        {
            v_pred_id.push_back(this->_connect_data[i].s_pre_id[j]);
            v_weight.push_back(this->_connect_data[i].weight[j]);
        }
    }
  
    //std::vector<int> stimulus_neuron_number;
    //std::vector<std::vector<double>> stimulus_data;
 
    cudaMalloc((void**)&p_nn, v_nn.size() * sizeof(int));
    cudaMalloc((void**)&p_pred_id, v_pred_id.size() * sizeof(int));
    cudaMalloc((void**)&p_weight, v_weight.size() * sizeof(double));

    cudaMalloc((void**)&this->p_cuda_spike_check, this->nn_neuron_num * sizeof(int));

    //cudaMemcpy(this->p_stim_number, (void*)&this->stimulus_neuron_number[0], stim_number * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(this->p_stim_data, (void*)&t_data[0], stim_length * stim_number *  sizeof(double),cudaMemcpyHostToDevice);
    //std::vector<int> stimulus_neuron_number;
    //std::vector<std::vector<double>> stimulus_data;

    cudaMemcpy(this->c_neuron, (void*)&this->_neuron_data[0], this->nn_neuron_num * sizeof(cuda_s_izkevich), cudaMemcpyHostToDevice);
    cudaMemcpy(p_nn, (void*)&v_nn[0], v_nn.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p_pred_id, (void*)&v_pred_id[0], v_pred_id.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p_weight, (void*)&v_weight[0], v_weight.size() * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(this->p_cuda_spike_check, (void*)&this->vt_spike_check[0], this->nn_neuron_num * sizeof(short int), cudaMemcpyHostToDevice);


}


void s_neuronal_netowrk::set_neuron_number(int nn)
{
    this->all_clear();
    this->nn_neuron_num = nn;
    this->_neuron_data.resize(nn);
    this->_connect_data.resize(nn);
   // this->_neuronal_xyz.resize(nn);
   // this->spike_color.resize(nn);

    cudaMalloc((void**)&this->c_neuron, nn * sizeof(cuda_s_izkevich));
    cudaMalloc((void**)&this->devStates, nn * sizeof(curandState));

    int n_block = (int)(nn / 1024) + 1;
    setup_rand << <n_block, 1024 >> > (this->devStates, (int)time(NULL), nn);

}


__global__ void cuda_fun_calcium(cuda_s_izkevich* v_neuron,  double dt, int time_step, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        double ica= 1.0 / (1 + exp(-1.1 * (v_neuron[tid].v + 34)));

        double dca = (5.8 * ica - (v_neuron[tid].calcium) * 0.01);

        v_neuron[tid].calcium += dt * dca;
    }
 
    return;
}

__global__ void cuda_fun_update_ca(cuda_s_izkevich* v_neuron, curandState* state,short int *spike_check,  double dt, int time_step, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        v_neuron[tid].exc_synapse_model(dt);
        v_neuron[tid].inh_synapse_model(dt);

        double tempu = v_neuron[tid].fun_u();
        double tempv = v_neuron[tid].fun_v();

        v_neuron[tid].u += tempu * dt;
        v_neuron[tid].v += (tempv + curand_normal_double(&state[tid]) * v_neuron[tid].noise_intensity) * dt;
        //v_neuron[tid].v +=       curand_normal_double(state); //(tempv  )* dt;

        v_neuron[tid].checking_spike(time_step);


        double ica = 1.0 / (1 + exp(-1.1 * (v_neuron[tid].v + 34)));

        double dca = (5.8 * ica - (v_neuron[tid].calcium) * 0.008);

        v_neuron[tid].calcium += dt * dca;

        v_neuron[tid].E_exc = 0.0;
        v_neuron[tid].E_inh = 0.0;



        if (v_neuron[tid].spike_checking == true)
        {
            spike_check[tid] = 1;
        }
        else
        {
            spike_check[tid] = 0;
        }

    }
}

__global__ void cuda_fun_synaptic_update(cuda_s_izkevich* v_neuron, curandState* state, int* p_nn, int* p_pre_id, double* p_weight, int size, int current_time, s_stdp_param stdp_param)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        int st, ft;
        if (tid == 0)
            st = 0;
        else
            st = p_nn[tid - 1];

        ft = p_nn[tid];


        for (int j = st; j < ft; j++)
        {

            p_weight[j] /= 1.0;


        }
    }
}

__global__ void cuda_fun_stdp_update(cuda_s_izkevich* v_neuron, curandState* state, int* p_nn, int* p_pre_id, double* p_weight, int size, int current_time ,s_stdp_param stdp_param)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        int st, ft;
        if (tid == 0)
            st = 0;
        else
            st = p_nn[tid - 1];

       ft = p_nn[tid];

       int pre_spike_time = v_neuron[tid].spiking_time;

       //if (v_neuron[tid].check_inh == false)
       {
           for (int j = st; j < ft; j++)
           {
               int kid = p_pre_id[j];
               int post_spike_time = v_neuron[kid].spiking_time;
               int dd = post_spike_time - pre_spike_time;
               
               if(abs(dd)<200)
                   p_weight[j] += stdp_param.p_rate * (stdp_param.w_max - p_weight[j]);
               if (current_time == post_spike_time || current_time == pre_spike_time)
               {
                   if (dd >= -150 && dd < 0)
                   {
                       //p_weight[j] -= stdp_param.n_rate * p_weight[j];
                   }

                   if (dd <= 1000 && dd >= 0)
                   {
                       p_weight[j] += stdp_param.p_rate * (stdp_param.w_max - p_weight[j]);
                   }

               }
           }
       }

      // else 
       {
           /*
           for (int j = st; j < ft; j++)
           {
               int kid = p_pre_id[j];
               int post_spike_time = v_neuron[kid].spiking_time;
               int dd = post_spike_time - pre_spike_time;

               if (current_time == post_spike_time || current_time == pre_spike_time)
               {

                   if (dd <= 5 && dd >= -5)
                   {
                       //   p_weight[j] += stdp_param.p_rate * (stdp_param.w_max - p_weight[j]);
                   }

               }
           }
           */
       }

    }
}


__global__ void cuda_fun_connect_update(cuda_s_izkevich* v_neuron, curandState* state, int* p_nn, int *p_pre_id, double * p_weight, int size)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        int st, ft;
      
        if (tid == 0)
            st = 0;
        else
            st = p_nn[tid - 1];

        ft = p_nn[tid];

                      
        if (v_neuron[tid].spike_checking == true)
        {
                    if (v_neuron[tid].check_inh == true)
                        for (int j = st; j < ft; j++) 
                        {
                            int kid = p_pre_id[j];
                            atomicAdd(&v_neuron[kid].E_inh, p_weight[j]);
                        }
                           
                    else

                        for (int j = st; j < ft; j++)
                        {
                            int kid = p_pre_id[j];
                            atomicAdd(&v_neuron[kid].E_exc, p_weight[j]);
                        }
        }
    
    }

}

__global__ void cuda_fun_stim_update(cuda_s_izkevich* v_neuron, int size, int stim_time,int real_time, int* p_stim_number, double* p_stim_data, short int* spike_check,int stim_number, int stim_length)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
   
    if (tid < stim_number)
    {

            if (stim_length>stim_time)
            {

                int _qq = p_stim_number[tid];
                if (p_stim_data[(tid)*stim_length+stim_time] > 0.5)
                {
                    v_neuron[_qq].spike_checking = true;
                    v_neuron[_qq].spiking_time = real_time;
                    spike_check[_qq] = 1;
                    
                }

            }

    }

}


__global__ void cuda_fun_ca_save(cuda_s_izkevich* v_neuron, double *p_ca_data, int time_data,int ca_ll ,int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        p_ca_data[time_data * size + tid] = v_neuron[tid].calcium;
    }
}

//__global__ void cuda_fun_internal_update(cuda_s_izkevich* v_neuron, curandState* state, short int* spike_check, double dt, int time_step, int size)


__global__ void cuda_fun_internal_update(cuda_s_izkevich* v_neuron, curandState* state, int* p_nn, int* p_pre_id, double* p_weight, int size,double internal_state)
//..__global__ void cuda_fun_internal_update(cuda_s_izkevich* v_neuron,double internal_state, double dt,  int size)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        if (tid >= 0 && tid < 300)
        {

            int st, ft;

            if (tid == 0)
                st = 0;
            else
                st = p_nn[tid - 1];

            ft = p_nn[tid];


            for (int j = st; j < ft; j++)
            {
                int kid = p_pre_id[j];
                p_weight[j] +=  (1.5+internal_state - p_weight[j]) * 0.1;//nternal_state;
                //p_weight[j] = (1.5 + internal_state);
                //atomicAdd(&v_neuron[kid].E_inh, p_weight[j]);
            }

             
           // v_neuron[tid].thre += (-0.01 * (v_neuron[tid].thre - 50) + internal_state) * dt;
        }
    }

}

__global__ void cuda_fun_update(cuda_s_izkevich* v_neuron, curandState* state, short int* spike_check,  double dt, int time_step, int size)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
 
       v_neuron[tid].exc_synapse_model(dt);
       v_neuron[tid].inh_synapse_model(dt);


        double tempu = v_neuron[tid].fun_u();
        double tempv = v_neuron[tid].fun_v();

        
        v_neuron[tid].u += tempu * dt;
        v_neuron[tid].v += (tempv + curand_normal_double(&state[tid]) * v_neuron[tid].noise_intensity) * dt;
        //v_neuron[tid].v +=       curand_normal_double(state); //(tempv  )* dt;
        
        v_neuron[tid].checking_spike(time_step);

        v_neuron[tid].E_exc = 0.0;
        v_neuron[tid].E_inh = 0.0;

        if (v_neuron[tid].spike_checking == true)
        {
            spike_check[tid] = 1;
        }
        else
        {
            spike_check[tid] = 0;
        }
                      
    }

}

__global__ void setup_exc_inh(double* exc, double* inh,  int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        exc[tid] = 0.0;
        inh[tid] = 0.0;

    }
}


void making_neuron_connection_list(std::vector<s_neuron_connection>& in_data, int nn)
{

    in_data.clear();
    in_data.resize(nn);

    std::random_device rd;
    
    std::mt19937 mer(rd());
    std::uniform_real_distribution<double> die(0.0,1.0 );
    for (int i = 0; i < nn; i++)
    {
        for (int j = 0; j < nn; j++)
        {
            if (die(mer) < 0.1)
            {
                in_data[i].s_pre_id.push_back(j);
                in_data[i].weight.push_back(0.14);
            }
        }
    }

}


/*
void GPU_test_ca(int nn, double dt, int st, int ft, double nois_intensity, std::vector<cuda_s_izkevich>& t_neuron, std::vector<s_neuron_connection>& connect_data,  std::vector<int>& ca_in_number, std::vector<std::vector<double>>& ca_out_data ,int ca_dd)
{

    cuda_s_izkevich* c_neuron;
    cudaMalloc((void**)&c_neuron, nn * sizeof(cuda_s_izkevich));

    curandState* devStates;
    cudaMalloc((void**)&devStates, nn * sizeof(curandState));

    int n_block = (int)(nn / 1024) + 1;
    setup_rand << <n_block, 1024 >> > (devStates, (int)time(NULL), nn);


    std::vector<double> v_exc(nn, 0.0);
    std::vector<double> v_inh(nn, 0.0);


    ca_out_data.clear();
    ca_out_data.resize(ca_in_number.size());

    for (int i = st; i < ft; i++)
    {
        for (int ii = 0; ii < nn; ii++)
        {
            v_exc[ii] = 0.0;
            v_inh[ii] = 0.0;
        }

      

        if (i % ca_dd == 0)
        {
            for (int j = 0; j < ca_in_number.size(); j++)
            {
                ca_out_data[j].push_back(t_neuron[ca_in_number[j]].calcium);
            }
        }

        cudaDeviceSynchronize();
        cudaMemcpy(c_neuron, (void*)&t_neuron[0], nn * sizeof(cuda_s_izkevich), cudaMemcpyHostToDevice);

        cuda_fun_update_ca << <n_block, 1024 >> > (c_neuron, devStates, nois_intensity, dt, i, nn);
        cudaDeviceSynchronize();

      //  cuda_fun_calcium << <n_block, 1024 >> > (c_neuron, dt, i, nn);
        //cudaDeviceSynchronize();

        cudaMemcpy((void*)&t_neuron[0], (void*)&c_neuron[0], nn * sizeof(cuda_s_izkevich), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();


        int thread_num = 1;
        int dnn = nn / thread_num;



        auto yy = [dnn, nn, &t_neuron, &v_exc, &v_inh, &connect_data](int pp)
        {
            for (int j = dnn * pp; j < dnn * (pp + 1); j++)
            {

                if (t_neuron[j].spike_checking == true)
                {
                    if (t_neuron[j].check_inh == true)

                        for (int k = 0; k < connect_data[j].s_pre_id.size(); k++)

                            v_inh[connect_data[j].s_pre_id[k]] += connect_data[j].weight[k];

                    else

                        for (int k = 0; k < connect_data[j].s_pre_id.size(); k++)

                            v_exc[connect_data[j].s_pre_id[k]] += connect_data[j].weight[k];


                }
            }
        };

        yy(0);

        //boost::thread_group tg;
        //for (int k = 0; k < thread_num; k++)  tg.add_thread(new boost::thread(std::bind(yy, k)));
        //tg.join_all();

        for (int tid = 0; tid < nn; tid++)
        {
            t_neuron[tid].E_exc = v_exc[tid];
            t_neuron[tid].E_inh = v_inh[tid];
        }

    }

    cudaFree(c_neuron);
    cudaFree(devStates);

}
*/



void GPU_internal_clock_test(s_neuronal_netowrk* nn_data)
{

    int nn = nn_data->nn_neuron_num;
    int n_thread = 500;
    int n_block = (int)(nn / n_thread) + 1;

    if (nn_data->calcium_recording == true)
    {
        nn_data->ca_ll = (nn_data->run_param.ft - nn_data->run_param.st) / nn_data->calcium_dd;
        cudaMalloc((void**)&nn_data->p_ca_data, nn * nn_data->ca_ll * sizeof(double));
        //nn_data->out_ca_data.resize(nn_data->nn_neuron_num, ll, false);
    }


    int ca_count = 0;

    double internal_state1 = 0;
    double internal_state2 = 0;
    double gamma1 = nn_data->internal_param.gamma1;
    double gamma2 = nn_data->internal_param.gamma2;

    double t_weight = 1.5;
    for (int i = nn_data->run_param.st; i < nn_data->run_param.ft; i++)
    {

        //internal state
                 
         nn_data->interal_state_1.push_back(t_weight);
         nn_data->interal_state_2.push_back(internal_state2);
         nn_data->interal_state_3.push_back(internal_state1);
         t_weight += (1.5 + internal_state2/3 - t_weight) * 0.1;

         double input = 0;

         if (i % 40000 == 0 && i>85000) 
         {
                input = 5.0;
         }

         internal_state1 +=(-gamma1*internal_state1)* nn_data->run_param.dt +input;
         internal_state2 += (-gamma2 * internal_state2 + internal_state1) * nn_data->run_param.dt;

            //internal_state1 = 7;
            //internal_state2 = 3;

            
         cuda_fun_internal_update << <n_block, n_thread >> > (nn_data->c_neuron, nn_data->devStates, nn_data->p_nn, nn_data->p_pred_id, nn_data->p_weight, nn, internal_state2/3.0);
        // cuda_fun_internal_update << <n_block, n_thread >> > (nn_data->c_neuron, internal_state, nn_data->run_param.dt, nn);

        
        




        if (nn_data->run_param.run_stim == true)
        {
            cuda_fun_stim_update << <n_block, n_thread >> > (nn_data->c_neuron, nn_data->nn_neuron_num, nn_data->stim_time, i, nn_data->p_stim_number, nn_data->p_stim_data, nn_data->p_cuda_spike_check, nn_data->stim_number, nn_data->stim_length);

            nn_data->stim_time++;  
            if (nn_data->stim_time >= nn_data->stimulus_data[0].size())
            {
                nn_data->stim_time = 0.0;
            }

            cudaDeviceSynchronize();
        }


        cuda_fun_connect_update << <n_block, n_thread >> > (nn_data->c_neuron, nn_data->devStates, nn_data->p_nn, nn_data->p_pred_id, nn_data->p_weight, nn);
        cudaDeviceSynchronize();



        if (nn_data->run_param.run_stdp == true)
        {
            cuda_fun_stdp_update << <n_block, n_thread >> > (nn_data->c_neuron, nn_data->devStates, nn_data->p_nn, nn_data->p_pred_id, nn_data->p_weight, nn, i, nn_data->stdp_param);
            cudaDeviceSynchronize();
        }



        if (nn_data->run_param.spiking_time_record == true)
        {

            cudaMemcpy((void*)&nn_data->vt_spike_check[0], (void*)&nn_data->p_cuda_spike_check[0], nn * sizeof(short int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            for (int ii = 0; ii < nn; ii++)            {
                if (nn_data->vt_spike_check[ii] == 1)
                {
                    s_xy_data t;
                    t.x = ii;
                    t.y = i;

                    nn_data->out_data.push_back(t);
                }
            }

        }

        if (nn_data->calcium_recording == true)
        {
            //__global__ void cuda_fun_ca_save(cuda_s_izkevich* v_neuron, double* p_ca_data, int time_data, int ca_ll, int size)
            if (i % nn_data->calcium_dd == 0)
            {
                cuda_fun_ca_save << <n_block, n_thread >> > (nn_data->c_neuron, nn_data->p_ca_data, ca_count, nn_data->ca_ll, nn);
                //cuda_s_izkevich* v_neuron, double* p_ca_data, int time_data, int size)
                ca_count++;
            }
            //if (i % nn_data->calcium_dd == 0)
            {
                //  for (int ii = 0; ii < nn; ii++)
                {
                    //    nn_data->out_ca_data(ii,ca_count) = nn_data->_neuron_data[ii].calcium;
                }
                // ca_count++;
            }
        }

        if (nn_data->calcium_recording == false)
        {
            cuda_fun_update << <n_block, n_thread >> > (nn_data->c_neuron, nn_data->devStates, nn_data->p_cuda_spike_check, nn_data->run_param.dt, i, nn);
        }
        else
        {
            cuda_fun_update_ca << <n_block, n_thread >> > (nn_data->c_neuron, nn_data->devStates, nn_data->p_cuda_spike_check, nn_data->run_param.dt, i, nn);
        }

        cudaDeviceSynchronize();

    }


    if (nn_data->calcium_recording == true)
    {
        nn_data->out_ca_data.clear();
        nn_data->out_ca_data.resize(nn * nn_data->ca_ll, 0.0);
        cudaMemcpy((void*)&nn_data->out_ca_data[0], (void*)&nn_data->p_ca_data[0], nn * nn_data->ca_ll * sizeof(double), cudaMemcpyDeviceToHost);
    }

    cudaMemcpy((void*)&nn_data->_neuron_data[0], (void*)&nn_data->c_neuron[0], nn * sizeof(cuda_s_izkevich), cudaMemcpyDeviceToHost);


    // nn_data->p_weight
    cudaMemcpy((void*)&nn_data->v_weight[0], (void*)&nn_data->p_weight[0], nn_data->v_weight.size() * sizeof(double), cudaMemcpyDeviceToHost);


    cudaDeviceSynchronize();


}




void GPU_STDP_stimulus_test(s_neuronal_netowrk* nn_data)
{

    int nn = nn_data->nn_neuron_num;
    int n_thread = 500;
    int n_block = (int)(nn / n_thread) + 1;

    //v_weight->p_weight;
     cudaMemcpy((void*)&nn_data->p_weight[0], (void*)&nn_data->v_weight[0], nn_data->v_weight.size() * sizeof(double), cudaMemcpyHostToDevice);

    if (nn_data->calcium_recording == true)
    {
         nn_data->ca_ll = (nn_data->run_param.ft - nn_data->run_param.st) / nn_data->calcium_dd;
         cudaMalloc((void**)&nn_data->p_ca_data, nn *nn_data->ca_ll* sizeof(double));
        //nn_data->out_ca_data.resize(nn_data->nn_neuron_num, ll, false);
    }


    int ca_count = 0;

    for (int i = nn_data->run_param.st; i < nn_data->run_param.ft; i++)
    {

        if (nn_data->run_param.run_stim == true)
        {
            cuda_fun_stim_update << <n_block, n_thread >> > (nn_data->c_neuron, nn_data->nn_neuron_num, nn_data->stim_time, i, nn_data->p_stim_number, nn_data->p_stim_data, nn_data->p_cuda_spike_check,nn_data->stim_number, nn_data->stim_length);

            nn_data->stim_time++;
            if (nn_data->stim_time >= nn_data->stimulus_data[0].size())
            {
                nn_data->stim_time = 0.0;
            }

            cudaDeviceSynchronize();
        }


        cuda_fun_connect_update << <n_block, n_thread >> > (nn_data->c_neuron, nn_data->devStates, nn_data->p_nn, nn_data->p_pred_id, nn_data->p_weight, nn);
        cudaDeviceSynchronize();



        if (nn_data->run_param.run_stdp == true)
        {
            cuda_fun_stdp_update << <n_block, n_thread >> > (nn_data->c_neuron, nn_data->devStates, nn_data->p_nn, nn_data->p_pred_id, nn_data->p_weight, nn, i, nn_data->stdp_param);
            cudaDeviceSynchronize();
        }



        if (nn_data->run_param.spiking_time_record == true)
        {
            
            cudaMemcpy((void*)&nn_data->vt_spike_check[0], (void*)&nn_data->p_cuda_spike_check[0], nn* sizeof(short int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            for (int ii = 0; ii < nn; ii++)
            {
                if (nn_data->vt_spike_check[ii] == 1)
                {
                    s_xy_data t;
                    t.x = ii; //neuron index
                    t.y = i; //time

                    nn_data->out_data.push_back(t);
                }
            }

        }

        if (nn_data->calcium_recording == true)
        {
            //__global__ void cuda_fun_ca_save(cuda_s_izkevich* v_neuron, double* p_ca_data, int time_data, int ca_ll, int size)
            if (i % nn_data->calcium_dd == 0)
            {
                cuda_fun_ca_save << <n_block, n_thread >> > (nn_data->c_neuron, nn_data->p_ca_data, ca_count, nn_data->ca_ll, nn);
                //cuda_s_izkevich* v_neuron, double* p_ca_data, int time_data, int size)
                ca_count++;
            }
            //if (i % nn_data->calcium_dd == 0)
            {
              //  for (int ii = 0; ii < nn; ii++)
                {
                //    nn_data->out_ca_data(ii,ca_count) = nn_data->_neuron_data[ii].calcium;
                }
               // ca_count++;
            }
        }
      
        if (nn_data->calcium_recording == false)
        {
            cuda_fun_update << <n_block, n_thread >> > (nn_data->c_neuron, nn_data->devStates, nn_data->p_cuda_spike_check,  nn_data->run_param.dt, i, nn);
        }
        else
        {
           cuda_fun_update_ca << <n_block, n_thread >> > (nn_data->c_neuron, nn_data->devStates, nn_data->p_cuda_spike_check, nn_data->run_param.dt, i, nn);
        }
        
        cudaDeviceSynchronize();
    }



    if (nn_data->calcium_recording == true)
    {
        nn_data->out_ca_data.clear(); 
        nn_data->out_ca_data.resize(nn* nn_data->ca_ll, 0.0);
        cudaMemcpy((void*)&nn_data->out_ca_data[0], (void*)&nn_data->p_ca_data[0], nn *nn_data->ca_ll*sizeof(double), cudaMemcpyDeviceToHost);
    }

    cudaMemcpy((void*)&nn_data->_neuron_data[0], (void*)&nn_data->c_neuron[0], nn * sizeof(cuda_s_izkevich), cudaMemcpyDeviceToHost);

    
    //p_weight->nn_data;
    cudaMemcpy((void*)&nn_data->v_weight[0], (void*)&nn_data->p_weight[0], nn_data->v_weight.size() * sizeof(double), cudaMemcpyDeviceToHost);


    cudaDeviceSynchronize();


}


