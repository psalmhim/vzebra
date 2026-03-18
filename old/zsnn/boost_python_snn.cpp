#include <boost/python.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include "CUDA_DM_NEURON.cuh"
#include "CUDA_DM_RUN.cuh"
#include <random>
//#include "opengl_dis.h"
#include <ctime>
#include <array>

template<typename T> inline
std::vector< T > py_list_to_std_vector(const boost::python::object& iterable)
{
    return std::vector< T >(boost::python::stl_input_iterator< T >(iterable),
        boost::python::stl_input_iterator< T >());
};


template <class T> inline
boost::python::list std_vector_to_py_list(std::vector<T> vector) {
    typename std::vector<T>::iterator iter;
    boost::python::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        list.append(*iter);
    }
    return list;
};


struct s_spike_data
{
    int id;
    int spike_time;
};


struct run_record_weight 
{
    std::vector<int> neurnal_num;
    std::vector<int> connection_num;
    std::vector<std::vector<double>> weight_data;
};




struct boost_neuronal_netowrk :s_neuronal_netowrk
{
    run_record_weight r_weight;
    
    void fun_boost_sclae(boost::python::list sclae_data);

   //void set_internal_param(double p1, double p2);
    
    void set_record_weight(int xx, int yy)
    {
        this->r_weight.neurnal_num.push_back(xx);
        this->r_weight.connection_num.push_back(yy);
    }

    boost::python::list  get_record_weight()
    {
        boost::python::list tt_out_data;

        for (int i = 0; i < this->r_weight.neurnal_num.size(); i++)
        {
            boost::python::list t_out_data;

            for (int j = 0; j < this->r_weight.weight_data[i].size(); j++)
            {
                t_out_data.append(this->r_weight.weight_data[i][j]);
            }

            tt_out_data.append(t_out_data);
        }
       
        return tt_out_data;
    }

    int get_stim_number()
    {
        return this->stim_number;
    }

    int get_stim_length()
    {
        return this->stim_length;
    }

    void set_stimulus(boost::python::list& neuron_num, boost::python::list& stimulus)
    {
        auto ll = boost::python::len(neuron_num);
        this->stimulus_neuron_number.clear();
        this->stimulus_data.clear();

        for (int i = 0; i < ll; i++)
        {
            this->stimulus_neuron_number.push_back(boost::python::extract<int>(neuron_num[i]));
        }


        this->stimulus_data.resize(ll);



        for (int i = 0; i < ll; i++)
        {
            boost::python::list _data = boost::python::extract<boost::python::list>(stimulus[i]);
            auto pp = boost::python::len(_data);

            for (int j = 0; j < pp; j++)
            {
                this->stimulus_data[i].push_back(boost::python::extract<double>(_data[j]));
            }

        }

        this->stim_number =(int)this->stimulus_neuron_number.size();


        if (this->stimulus_data.size() > 0)
        {
            this->stim_length = (int)this->stimulus_data[0].size();
        }
        else
        {
            this->stim_length = 0;
        }


        std::vector<double> t_data;
        for (int i = 0; i < stimulus_data.size(); i++)
            for (int j = 0; j < stimulus_data[i].size(); j++)
                t_data.push_back(stimulus_data[i][j]);


        cudaFree(p_stim_number);
        cudaFree(p_stim_data);

        cudaMalloc((void**)&p_stim_number, stim_number * sizeof(int));
        cudaMalloc((void**)&p_stim_data, stim_length * stim_number * sizeof(double));

        cudaMemcpy(this->p_stim_number, (void*)&this->stimulus_neuron_number[0], stim_number * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(this->p_stim_data, (void*)&t_data[0], stim_length * stim_number * sizeof(double), cudaMemcpyHostToDevice);
    
    }
     
    void run_display();

    void cpu_run(std::vector<s_spike_data>& spike_data);

    boost::python::list  get_spike_data()
    {
        
        boost::python::list _out_data;
        if (this->run_param.spiking_time_record == true)
        {
            auto ll = (this->out_data.size());
            
            for (int i = 0; i < ll; i++)
            {
                boost::python::list t_out_data;
            
                t_out_data.append(this->out_data[i].x);
                t_out_data.append(this->out_data[i].y);
                _out_data.append(t_out_data);
            }

            
        }

        return _out_data;
    }

    boost::python::list  get_ca_data() 
    {
        boost::python::list ca_out_data;
        
        if (this->calcium_recording == true)
        {
            for (int i = 0; i < this->ca_ll; i++)
            {
                boost::python::list t_out_data;

                for (int j = 0; j< this->nn_neuron_num; j++)
                {
                    t_out_data.append(this->out_ca_data[i * this->nn_neuron_num + j]);
                }

                ca_out_data.append(t_out_data);

            }
        }

        return ca_out_data;
    }

    void  cuda_internal_run_python()
    {
               
        //std::vector<s_xy_data> out_data;
        this->cuda_internal_run_stdp();
    }

    void  cuda_run_python()
    {
        //std::vector<s_xy_data> out_data;
        this->cuda_run_stdp();
    }

    boost::python::list get_internal_state1()
    {
        boost::python::list _out_data;

        for (int i = 0; i < this->interal_state_1.size(); i++)
        {
            _out_data.append(this->interal_state_1[i]);

        }
        return _out_data;
    }

    boost::python::list get_internal_state3()
    {
        boost::python::list _out_data;

        for (int i = 0; i < this->interal_state_3.size(); i++)
        {
            _out_data.append(this->interal_state_3[i]);

        }
        return _out_data;
    }

    boost::python::list get_internal_state2()
    {
        boost::python::list _out_data;

        for (int i = 0; i < this->interal_state_2.size(); i++)
        {
            _out_data.append(this->interal_state_2[i]);

        }
        return _out_data;
    }


    boost::python::list gpu_get_weight_matrix()
    {
        boost::python::list _out_data;

        for (int i = 0; i < this->nn_neuron_num; i++)
        {
            boost::python::list t_out_data;
            std::vector<double> tt(this->nn_neuron_num, 0.0);
            
            int st = 0;
            if (i - 1 < 0)
            {
                st = 0;
            }
           
            else
            {
                st = this->v_nn[i - 1];
            }


            for (int j = st; j < v_nn[i] ; j++)
            {
                tt[v_pred_id[j]] = this->v_weight[j];
                //tt[ii] = this->_connect_data[i].weight[j];
            }

            for (int j = 0; j < this->nn_neuron_num; j++)
            {
                t_out_data.append(tt[j]);
            }

            _out_data.append(t_out_data);
        }

        return _out_data;

    }


    boost::python::list get_weight_matrix()
    {
        boost::python::list _out_data;

        for (int i = 0; i < this->nn_neuron_num; i++)
        {
            boost::python::list t_out_data;

            std::vector<double> tt(this->nn_neuron_num,0.0);
        
            

            for (int j = 0; j < this->_connect_data[i].s_pre_id.size(); j++)
            {
                int ii = this->_connect_data[i].s_pre_id[j];
                                    
                tt[ii] = this->_connect_data[i].weight[j];
            }

            for (int j = 0; j < this->nn_neuron_num; j++)
            {
                t_out_data.append(tt[j]);
            }

            _out_data.append(t_out_data);
        }

        return _out_data;
    }

    void python_cpu();
};

void boost_neuronal_netowrk::fun_boost_sclae(boost::python::list scale_data)
{

    std::vector<double> v_scale_data;

    for (int i = 0; i < boost::python::len(scale_data); i++)
    {

        v_scale_data.push_back(boost::python::extract<double >(scale_data[i]));
    }


    //this->fun_calculate_scale(v_scale_data);
}


void boost_neuronal_netowrk::python_cpu()
{

    int nn = this->nn_neuron_num;
    std::mt19937 gen((unsigned int)time(NULL)+ this->run_param.st);
    std::normal_distribution<> norm(0, 1);
    std::uniform_real_distribution<> uni_dist(0.0, 1.0);

   int stim_time = 0;

    this->r_weight.weight_data.clear();

    for (int i = 0; i < this->r_weight.neurnal_num.size(); i++)
    {
        this->r_weight.weight_data.resize(this->r_weight.neurnal_num.size());
    }
    
    //this->out_data.clear();

    
    for (int current_time = this->run_param.st; current_time < this->run_param.ft; current_time++)
    {

        if (this->run_param.spiking_time_record == true)
        {
            for (int i = 0; i < nn; i++)
            {
                if (this->_neuron_data[i].spike_checking == true)
                {
                    s_xy_data tt;
                    tt.x = i;
                    tt.y = (int)this->_neuron_data[i].spiking_time;
                    this->out_data.push_back(tt);
                }
            }
        }


        for (int i = 0; i < nn; i++)
        {
            this->_neuron_data[i].cpu_exc_synapse_model(this->run_param.dt);
            this->_neuron_data[i].cpu_inh_synapse_model(this->run_param.dt);

            double tempu = this->_neuron_data[i].cpu_fun_u();
            double tempv = this->_neuron_data[i].cpu_fun_v();

            this->_neuron_data[i].u += tempu * this->run_param.dt;
            this->_neuron_data[i].v += (tempv + norm(gen) * this->_neuron_data[i].noise_intensity) * this->run_param.dt;

            this->_neuron_data[i].cpu_checking_spike(current_time);
        }


        for (int i = 0; i < nn; i++)
        {
            this->_neuron_data[i].E_exc = 0.0;
            this->_neuron_data[i].E_inh = 0.0;
        }


        if (this->run_param.run_stim == true)
        {

            for (int i = 0; i < this->stimulus_neuron_number.size(); i++)
            {

                int tn = stimulus_neuron_number[i];

                if (stim_time < this->stimulus_data[0].size())
                {
                    if (this->stimulus_data[i][stim_time] >= 0.5)
                    {
                        this->_neuron_data[tn].spiking_time = current_time;
                        this->_neuron_data[tn].spike_checking = true;
                    }
                }

            }

            stim_time++;
        }

        for (int i = 0; i < nn; i++)
        {
            if (this->_neuron_data[i].spike_checking == true)
            {
                if (this->_neuron_data[i].check_inh == true)
                {
                    for (int j = 0; j < this->_connect_data[i].s_pre_id.size(); j++)
                    {
                        int tn = this->_connect_data[i].s_pre_id[j];
                        this->_neuron_data[tn].E_inh += this->_connect_data[i].weight[j];
                    }

                }

                if (this->_neuron_data[i].check_inh == false)
                {
                    for (int j = 0; j < this->_connect_data[i].s_pre_id.size(); j++)
                    {
                        int tn = this->_connect_data[i].s_pre_id[j];
                        this->_neuron_data[tn].E_exc += this->_connect_data[i].weight[j];
                    }
                }
            }
        }


        for (int i = 0; i < this->r_weight.neurnal_num.size(); i++)
        {
            int ii = this->r_weight.neurnal_num[i];
            int jj = this->r_weight.connection_num[i];

            this->r_weight.weight_data[i].push_back(this->_connect_data[ii].weight[jj]);

        }


        if (this->run_param.run_stdp == true)
        {
            for (int i = 0; i < nn; i++)
            {
                int st1 =(int)_neuron_data[i].spiking_time;
                for (int j = 0; j < this->_connect_data[i].s_pre_id.size(); j++)
                {
                    int tn = this->_connect_data[i].s_pre_id[j];
                    int st2 =(int) _neuron_data[tn].spiking_time;
                    int dd = st2 - st1;

                    if (st1 == current_time || st2 == current_time)
                    {
                        if (dd >= 0 && dd <150)
                        {
                            this->_connect_data[i].weight[j] += 0.8 * (3.5 - this->_connect_data[i].weight[j]);
                        }

                        if (dd >= -150 && dd < 0)
                        {
                            this->_connect_data[i].weight[j] -= 0.8 * this->_connect_data[i].weight[j];
                        }

                    }
                }
            }

            if (current_time % 10000 == 0 && current_time>0) {

                for (int i = 0; i < nn; i++) {

                    double temp_sum = 0.0;
                    double ll = (double)this->_connect_data[i].weight.size();

                    for (int j = 0; j < this->_connect_data[i].weight.size(); j++)
                    {
                        //if (temp_sum <this->_connect_data[i].weight[j])
                        temp_sum += this->_connect_data[i].weight[j];
                    }
                    
                    //  temp_sum /= ll;


                    for (int j = 0; j < this->_connect_data[i].weight.size(); j++)
                    {
                       // this->_connect_data[i].weight[j] = this->_connect_data[i].weight[j] / temp_sum * 1.9;
                    }

                }
            }

 
        }



    }

    return;
}


void boost_neuronal_netowrk::cpu_run(std::vector<s_spike_data> &spike_data)
{

    int nn = 1000;

    std::vector<cpu_s_izkevich> v_neuron(nn);
    
    std::vector<std::vector<int>> connect_data(nn);

    std::vector<std::vector<double>> connect_data_weight(nn);

    int st = 0;
    int ft = 20000*50;

    double dt = 0.05;
    double noise_intensity=10.5;

    std::mt19937 gen((unsigned int)time(NULL));
    std::normal_distribution<> norm(0,1);
    std::uniform_real_distribution<> uni_dist(0.0, 1.0);

    std::vector<std::vector<int>> r_spike_data(nn);
    std::vector<int> firing_rate(nn, 0);



    //connectivity 
    for (int i = 0; i < nn; i++)
    {
        for (int j = i+1; j < nn; j++)
        {
            if (uni_dist(gen) < 0.01)
            {
                if (uni_dist(gen) < 0.5)
                {
                    connect_data[i].push_back(j);
                    connect_data_weight[i].push_back(uni_dist(gen)*5.65);
                }
                else
                {
                    connect_data[j].push_back(i);
                    connect_data_weight[j].push_back(uni_dist(gen) * 5.65);
                }
            }
        }
    }


    spike_data.clear();

    for (int current_time = st; current_time < ft; current_time++)
    {

        for (int i = 0; i < nn; i++)
        {
            v_neuron[i].exc_synapse_model(dt);
            v_neuron[i].inh_synapse_model(dt);

            double tempu = v_neuron[i].fun_u();
            double tempv = v_neuron[i].fun_v();

            v_neuron[i].u += tempu * dt;
            v_neuron[i].v += (tempv + norm(gen) * noise_intensity) * dt;

            v_neuron[i].checking_spike(current_time);
        }



        for (int i = 0; i < nn; i++)
        {
            if (v_neuron[i].spike_checking == true)
            {
                s_spike_data ss;
                ss.id = i;
                ss.spike_time = current_time;
                spike_data.push_back(ss);

                r_spike_data[i].push_back(current_time);

            }
        }



        if (current_time%2500==0 && current_time>20000)
        {
            for (int i = 0; i < nn; i++)
            {
                int temp_sum = 0;
                for (int j = 0; j < r_spike_data[i].size(); j++)
                {
                   if ( r_spike_data[i][j] >= current_time-20000)
                   {
                         temp_sum++;
                   }
                }
                firing_rate[i] = temp_sum;
            
                
            }


            
            for (int i = 0; i < nn; i++)
            {

                if (firing_rate[i] > 8)
                {
                    for (int j = 0; j < connect_data_weight[i].size(); j++)
                    {
                        connect_data_weight[i][j] *= 0.9;
                    }
                }

                if (firing_rate[i] < 2)
                {
                    for (int j = 0; j < connect_data_weight[i].size(); j++)
                    {
                        connect_data_weight[i][j] *= 1.1;
                    }
                }

            }

                    

        }


        for (int i = 0; i < nn; i++)
        {
            v_neuron[i].E_exc = 0.0;
            v_neuron[i].E_inh = 0.0;
        }



        for (int i = 0; i < nn; i++)
        {

            for (int j = 0; j < connect_data[i].size(); j++)
            {
                int nn = connect_data[i][j];
                if (v_neuron[nn].spike_checking == true)
                {
                    v_neuron[i].E_exc += connect_data_weight[i][j];
                }
            }
        }

    }


    return;

}

char const* greet()
{
    return "Monet_SNN_DM, world wow";
}

using namespace boost::python;


struct boost_diffusion
{
    char const* get_vestion()
    {
        return "Monet_DM_diffustion, world wow";
    }


    boost::python::list run(double dt, int T, boost::python::list iinit, boost::python::list connect_data, double g,double gamma)
    {
               

        int num_neuron =(int)boost::python::len(iinit);

        std::vector<double> init_data(num_neuron);

               
        //ininiaization 
        for (int i = 0; i < num_neuron; i++)
        {
            double  tt = boost::python::extract<double>(iinit[i]);

            init_data[i] = tt;
            
        }

        
        //ininiaization connect_data;
        std::vector<std::vector<int>> con_data(num_neuron);

        for (int i = 0; i < boost::python::len(connect_data); i++)
        {
            boost::python::list tt = boost::python::extract<boost::python::list>(connect_data[i]);

            int t1 = boost::python::extract<int>(tt[0]);
            int t2 = boost::python::extract<int>(tt[1]);

            con_data[t1].push_back(t2);
        }

                
        //iteration;
        std::vector<double> output_data(num_neuron * T);
     

        for (int i = 0; i < T; i++)
        {

            std::vector<double> couplig_data(num_neuron, 0.0);

            for (int j = 0; j < num_neuron; j++)
            {
                int ii =  j * T + i ;

                output_data[ii] = init_data[j];
            
            }

            for (int j = 0; j < num_neuron; j++)
            {
                double t_sum = 0;
                for (int k = 0; k < con_data[j].size(); k++)
                {

                    int pp = con_data[j][k];
                    t_sum += init_data[j] - init_data[pp];

                }

                couplig_data[j] = t_sum;
            }

            for (int j = 0; j < num_neuron; j++)
            {
               // double du = fu(init_data[(int)2 * j], init_data[(int)2 * j + 1]);
                //double dv = fv(init_data[(int)2 * j], init_data[(int)2 * j + 1]);

                init_data[j] += (-gamma* init_data[j] - g * couplig_data[j]) * dt;
              
            }


        }

        boost::python::list out_python;


        for (int i = 0; i < T; i++)
        {
            boost::python::list t_list;

            for (int j = 0; j < num_neuron; j++)
            {

                int ii =  j * T + i;
                double tx = output_data[ii];
               

                t_list.append(tx);
            }

            out_python.append(t_list);

        }

        return out_python;

    }


};


struct boost_FHN
{
    char const* get_vestion()
    {
       return "Monet_DM_FHN, world wow";
    }

    const std::array<double, 2>  fun_fitznagumo(std::array<double, 2>& input_data)
    {

        std::array<double, 2>  output_data;
        
        output_data[0] = (input_data[0] - input_data[0] * input_data[0] * input_data[0] / 3.0 + input_data[1]) * 3.0;
        output_data[1] = (-input_data[0] + 0.7 - 0.8 * input_data[1]) / 3.0;

    }
    

    boost::python::list run(double dt, int T, boost::python::list iinit , boost::python::list& connect_data,double g)
    {
        int num_neuron =(int)boost::python::len(iinit);

        std::vector<double> init_data(2* num_neuron);

        for (int i = 0; i < num_neuron; i++)
        { 
            boost::python::list tt=boost::python::extract<boost::python::list>(iinit[i]);

            double t1=boost::python::extract<double>(tt[0]);
            double t2=boost::python::extract<double>(tt[1]);

            init_data[(int)2*i] = t1;
            init_data[(int)2*i+1] = t2;
        }
        

        std::vector<std::vector<int>> con_data(num_neuron);

        for (int i = 0; i < boost::python::len(connect_data); i++)
        {
            boost::python::list tt = boost::python::extract<boost::python::list>(connect_data[i]);

            int t1 = boost::python::extract<int>(tt[0]);
            int t2 = boost::python::extract<int>(tt[1]);


            con_data[t1].push_back(t2);
        }




        auto fu = [](double &u, double &v) ->double { return (u - u * u * u / 3.0 + v) * 3.0; };
        auto fv = [](double &u, double &v) ->double { return  (-u + 0.7 - 0.8 * v) / 3.0;  };

                  
        std::vector<double> output_data(2 * num_neuron*T);
      
        

        for (int i = 0; i < T; i++)
        {

            std::vector<double> couplig_data(num_neuron, 0.0);

            for (int j = 0; j < num_neuron; j++)
            {
                int ii = 2 * j * T + i * 2;

                output_data[ii] = init_data[(int)2 * j];
                output_data[ii+1] = init_data[(int)2 * j+1];
            }

            
            for (int j = 0; j < num_neuron; j++)
            {
                double t_sum = 0;
                for (int k = 0; k < con_data[j].size(); k++)
                {

                    int pp = con_data[j][k];
                    t_sum+=init_data[(int)2 * j] - init_data[(int)2 * pp];

                }

                couplig_data[j] = t_sum;
            }

            for (int j = 0; j < num_neuron; j++)
            {
                double du=fu(init_data[(int)2*j], init_data[(int)2*j+1]);
                double dv=fv(init_data[(int)2*j], init_data[(int)2*j+1]);
                
                init_data[(int)2 * j] += (du-1.2-g*couplig_data[j]) * dt;
                init_data[(int)2 * j+1] += dv * dt;
            }


        }

        boost::python::list out_python;

        
        for (int i = 0; i < T; i++)
        {
            boost::python::list t_list;
         
            for (int j = 0; j < num_neuron; j++)
            {

                int ii = 2 * j * T + i * 2;
                double tx = output_data[ii];
                double ty = output_data[ii + 1];

                t_list.append(tx);
            }

            out_python.append(t_list);

        }

        return out_python;

    }


};

BOOST_PYTHON_MODULE(MONET_SNN_CUDA_PYTHON)
{
    def("greet", greet);

    class_<boost_FHN>("FHN")
        .def("get_verstion",&boost_FHN::get_vestion)
        .def("run", &boost_FHN::run);
    
    class_<boost_diffusion>("diffusion")
        .def("get_verstion", &boost_diffusion::get_vestion)
        .def("run", &boost_diffusion::run);

    class_<boost_neuronal_netowrk>("DM_SNN")
        .def("all_clear", &boost_neuronal_netowrk::all_clear)
        .def("set_neuron_number", &boost_neuronal_netowrk::set_neuron_number)
        .def("get_neuron_number", &boost_neuronal_netowrk::get_neuron_number)
        .def("set_inhibtion_neuron", &boost_neuronal_netowrk::set_inhibtion_neuron)
        .def("get_inhibtion_neuron", &boost_neuronal_netowrk::get_inhibtion_neuron)
        .def("set_connection", &boost_neuronal_netowrk::set_connection)
        .def("set_modify_connection",&boost_neuronal_netowrk::set_modify_connection)
        .def("set_neuron_xyz", &boost_neuronal_netowrk::set_neuron_xyz)
        .def("set_stdp_param", &boost_neuronal_netowrk::set_stdp_param)
        .def("set_run_param", &boost_neuronal_netowrk::set_run_param)
        .def("set_stimulus", &boost_neuronal_netowrk::set_stimulus)
        .def("set_record_weight", &boost_neuronal_netowrk::set_record_weight)
        .def("set_noise", &boost_neuronal_netowrk::set_noise)
        .def("get_version", &boost_neuronal_netowrk::get_version)
        .def("get_spike_data", &boost_neuronal_netowrk::get_spike_data)
        .def("get_ca_data", &boost_neuronal_netowrk::get_ca_data)
        .def("get_record_weight", &boost_neuronal_netowrk::get_record_weight)
        .def("get_stim_number", &boost_neuronal_netowrk::get_stim_number)
        .def("get_stim_length", &boost_neuronal_netowrk::get_stim_length)
        .def("get_weight_matrix", &boost_neuronal_netowrk::get_weight_matrix)
        .def("gpu_get_weight_matrix", &boost_neuronal_netowrk::gpu_get_weight_matrix)
        .def("cuda_run_python", &boost_neuronal_netowrk::cuda_run_python)
        .def("create_cuda_memory", &boost_neuronal_netowrk::create_cuda_memory)
        .def("run_display", &boost_neuronal_netowrk::run_display)
        .def("set_calcium_recording", &boost_neuronal_netowrk::set_calcium_recording)
        .def("cpu_run_python", &boost_neuronal_netowrk::python_cpu)
        .def("get_inetrnal_state1", &boost_neuronal_netowrk::get_internal_state1)
        .def("get_inetrnal_state2", &boost_neuronal_netowrk::get_internal_state2)
        .def("get_inetrnal_state3", &boost_neuronal_netowrk::get_internal_state3)
        .def("cuda_internal_run_python", &boost_neuronal_netowrk::cuda_internal_run_python)
        .def("set_internal_parm", &boost_neuronal_netowrk::set_internal_param)
        .def("fun_hebbian_rate", &boost_neuronal_netowrk::fun_hebbian_rate_spike)
        .def("fun_set_weight", &boost_neuronal_netowrk::fun_set_weight)
        ;

}


void  boost_neuronal_netowrk::run_display()
{ /*
    int argc = 0;
    char** argv;

    //set_intensity(noise);
    this->run_param.spiking_time_record = false;
    this->calcium_recording = false;

    srand(GetCurrentProcessId());
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return;
    }

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);

    int x = 20;
    int y = 10;
    glutInitWindowPosition(x, y);
    int win = glutCreateWindow("MONET SNN CORTICAL COLUM MODEL");
    printf("window id: %d\n", win);

    InitializeGlutCallbacks_1();

    // Must be done after glut is initialized!
    GLenum res = glewInit();
    if (res != GLEW_OK) {
        fprintf(stderr, "Error: '%s'\n", glewGetErrorString(res));
        return;
    }

    GLclampf Red = 0.0f, Green = 0.0f, Blue = 0.0f, Alpha = 0.0f;
    glClearColor(Red, Green, Blue, Alpha);

    //set_neural_network(this);
    //Create_neuronal_network2();
 
    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CW);
    glCullFace(GL_BACK);

    CompileShaders();
    glutMainLoop();

    
    */
}



