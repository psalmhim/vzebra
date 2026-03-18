#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>   
#include <vector>
#include "CUDA_DM_NEURON.cuh"
#include <cstdio>
//#include <boost/multi_array.hpp>
//#include "opengl_dis.h"
//#include "kernel.cuh"
//#include "ogldev_math_3d.h"

//#include <boost/numeric/ublas/matrix.hpp>

/*/
struct Vertex
{
    Vector3f pos;
    Vector3f color;

    Vertex() {}

    Vertex(float x, float y, float z)
    {
        pos = Vector3f(x, y, z);

        float red = (float)rand() / (float)RAND_MAX;
        float green = (float)rand() / (float)RAND_MAX;
        float blue = (float)rand() / (float)RAND_MAX;
        color = Vector3f(red, green, blue);
    }

    Vertex(float x, float y, float z, float red, float green, float blue)
    {
        pos = Vector3f(x, y, z);
        //float red = (float)rand() / (float)RAND_MAX;
        //float green = (float)rand() / (float)RAND_MAX;
        //float blue = (float)rand() / (float)RAND_MAX;
        color = Vector3f(red, green, blue);
    }

};

*/
struct s_xy_data
{
	int x;
	int y;
};

struct s_run_check
{
	bool spiking_time_record = false;
	bool run_stdp = false;
	bool run_stim = false;

	double dt = 0.05;
	int st;
	int ft;
	double noise_intensity;

};

__global__ void cuda_fun_update(cuda_s_izkevich* v_neuron, curandState* state, double noise_intensity, double dt, int time_step, int size);
//__global__ void setup_rand(curandState* state, int seed, int size);

struct s_neuronal_netowrk;

//void GPU_STDP_stimulus_test(s_neuronal_netowrk* nn_data, std::vector<s_xy_data>& out_data);

void GPU_STDP_stimulus_test(s_neuronal_netowrk* nn_data);

void GPU_internal_clock_test(s_neuronal_netowrk* nn_data);

void making_neuron_connection_list(std::vector<s_neuron_connection>& in_data, int nn);

struct s_internal_param
{
    double gamma1 = 0.0;
    double gamma2 = 0.0;

};


struct s_neuronal_netowrk
{
    s_internal_param internal_param;

    void set_internal_param(double p1, double p2)
    {
        this->internal_param.gamma1 = p1;
        this->internal_param.gamma2 = p2;

        return; 
    }



    short int* p_cuda_spike_check;

    std::vector<short int> vt_spike_check;
    
    std::vector<double> interal_state_1;
    std::vector<double> interal_state_2;
    std::vector<double> interal_state_3;

   

    void save_spike_data(char *file_name)
    {
        //FILE* file;

//        fopen_s(&file, file_name, "w");

        for (int i = 0; i < this->out_data.size(); i++)
        {
 //           fprintf(file, "%d %d\n", out_data[i].x, out_data[i].y);
        }

  //      fclose(file);
   //     return;
    }

    std::vector<double> out_ca_data;// (nn* ca_ll, 0.0);
    double *p_ca_data;
    int* p_stim_number;
    double* p_stim_data;

    int stim_number;
    int stim_length;

    int ca_ll;

    cuda_s_izkevich* c_neuron;
    //cudaMalloc((void**)&c_neuron, nn * sizeof(cuda_s_izkevich));

    curandState* devStates;
    //cudaMalloc((void**)&devStates, nn * sizeof(curandState));
    int* p_nn, * p_pred_id;
    double* p_weight;

    ~s_neuronal_netowrk()
    {
        this->all_clear();
    }

    //std::vector<Vertex> _neuronal_xyz;
    std::vector<s_neuron_connection> _connect_data;
    std::vector<cuda_s_izkevich> _neuron_data;

    //std::vector<Vertex> spike_color;


    std::vector<int> stimulus_neuron_number;
    std::vector<std::vector<double>> stimulus_data;


    s_stdp_param stdp_param;

    s_run_check run_param;


    std::vector<s_xy_data> out_data;
   // boost::numeric::ublas::matrix<double> out_ca_data;

    std::vector<int> v_nn;
    std::vector<int> v_pred_id;
    std::vector<double> v_weight;




    bool calcium_recording = false;
    int calcium_dd;
    
    void set_calcium_recording(int t_dd)
    {
        this->calcium_recording = true;
        this->calcium_dd =t_dd;

    }

    double intensity = 0.0;

    int nn_neuron_num = 0;

    int stim_time = 0;

    int r_st = 0;
    int r_ft = 0;
    void all_clear()
    {
        //this->_neuronal_xyz.clear();
        this->_connect_data.clear();
        this->_neuron_data.clear();

        this->stimulus_neuron_number.clear();
        this->stimulus_data.clear();

        this->out_data.clear();
        this->out_ca_data.clear();
        

        cudaFree(this->p_stim_number);
        cudaFree(this->p_stim_data);
        cudaFree(this->c_neuron);
        cudaFree(this->devStates);

        cudaFree(this->p_nn);
        cudaFree(this->p_pred_id);
        cudaFree(this->p_weight);

        cudaFree(this->p_ca_data);

        cudaFree(this->p_cuda_spike_check);

    }


    void set_stdp_param(double a, double b, double c)
    {
        this->stdp_param.w_max = a;
        this->stdp_param.p_rate = b;
        this->stdp_param.n_rate = c;
    }


    void set_neuron_number(int nn);
  

    int get_neuron_number()
    {
        return this->nn_neuron_num;
    }

    void set_inhibtion_neuron(int nn)
    {
        this->_neuron_data[nn].check_inh = true;
        this->_neuron_data[nn].a = 0.1;
        this->_neuron_data[nn].d = 2;
    }

    bool get_inhibtion_neuron(int nn)
    {
        return _neuron_data[nn].check_inh;
    }

    void set_modify_connection(int from, int to, double weight)
    {
        for (int i = 0; i < this->_connect_data[from].s_pre_id.size(); i++)
        {
            if (this->_connect_data[from].s_pre_id[i] == to)
            {
                //this->_connect_data[from].s_pre_id.push_back(to);
        
                this->_connect_data[from].weight[i]=weight;

            }
        }
    }


    void set_connection(int from, int to, double weight)
    {
        this->_connect_data[from].s_pre_id.push_back(to);
        this->_connect_data[from].weight.push_back(weight);
    }

    void set_neuron_xyz(int nn, float x, float y, float z, float r, float g, float b)
    {

       // this->_neuronal_xyz[nn] = Vertex(x, y, z, 0.1f, 0.1f, 0.1f);

        //this->spike_color[nn].color.r = r;
        //this->spike_color[nn].color.g = g;
        //this->spike_color[nn].color.b = b;


    }

    char const* get_version()
    {
        return "MONET SNN DM VERSION 1.5 2024.07.17";
    }

       

    void set_run_param(double dt, int st, int ft,  bool r_record, bool r_stdp, bool r_stim)
    {

        this->run_param.dt = dt;
        this->run_param.st = st;
        this->run_param.ft = ft;
       // this->run_param.noise_intensity = 0.0;// noise_intensity;
        this->run_param.spiking_time_record = r_record;
        this->run_param.run_stdp = r_stdp;
        this->run_param.run_stim = r_stim;

        this->r_st = st;
        this->r_ft = ft;

    }

    void set_noise(int i, double t_data)
    {

        this->_neuron_data[i].noise_intensity = t_data;
    }

    void cuda_internal_run_stdp()
    {
        this->stim_time = 0;
        GPU_internal_clock_test(this);
        GPU_STDP_stimulus_test(this);

    }


    void   cuda_run_stdp()
    {
        this->stim_time = 0;
        GPU_STDP_stimulus_test(this);
    
    }
    

    void create_cuda_memory();

    void fun_hebbian_rate_spike(int window);
    
    void fun_set_weight(int i, int j, double w);


};



//void GPU_test_ca(int nn, double dt, int st, int ft, double nois_intensity, std::vector<cuda_s_izkevich>& t_neuron, std::vector<s_neuron_connection>& connect_data, std::vector<int>& ca_in_number, std::vector<std::vector<double>>& ca_out_data, int ca_dd);
//void GPU_test(int nn, double dt, int st, int ft, double nois_intensity, std::vector<cuda_s_izkevich>& t_neuron);
//void GPU_test(int nn, double dt, int st, int ft, double nois_intensity, std::vector<cuda_s_izkevich>& t_neuron, std::vector<s_neuron_connection>& connect_data);
//void GPU_test(int nn, double dt, int st, int ft, double nois_intensity, std::vector<cuda_s_izkevich>& t_neuron, std::vector<s_neuron_connection>& connect_data, std::vector<int>& in_number, std::vector<std::vector<double>>& out_data, int dd);
//void GPU_STDP_test(int nn, double dt, int st, int ft, double nois_intensity, std::vector<cuda_s_izkevich>& t_neuron, std::vector<s_neuron_connection>& connect_data, std::vector<int>& in_number, std::vector<std::vector<double>>& out_data, int dd);
//void GPU_STDP_stimulus_test(int nn, double dt, int st, int ft, double nois_intensity, std::vector<cuda_s_izkevich>& t_neuron, std::vector<s_neuron_connection>& connect_data, std::vector<int>& stimulus_neuron_number, std::vector<std::vector<double>>& stimulus_data, std::vector<int>& in_number, std::vector<std::vector<double>>& out_data, int dd);
//void GPU_STDP_stimulus_test(int nn, double dt, int st, int ft, double nois_intensity, std::vector<cuda_s_izkevich>& t_neuron, std::vector<s_neuron_connection>& connect_data, std::vector<int>& stimulus_neuron_number, std::vector<std::vector<double>>& stimulus_data, std::vector<s_xy_data>& out_data);
//void GPU_test(int nn, double dt, int time_step, double nois_intensity, std::vector<double>& out_data);