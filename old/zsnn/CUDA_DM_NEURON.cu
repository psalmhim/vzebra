#include "CUDA_DM_NEURON.cuh"



/*
	__device__  double cuda_s_izkevich::fun_v()
	{
		return c1 * v * v + c2 * v + c3 - c4 * u + c5 * exc_synap * (v - 10.0) + c6 * inh_synap * (v + 60);
	}

	 __device__  double cuda_s_izkevich::fun_u()
	{
		return a * (b * v - u);
	}

	 __device__ void cuda_s_izkevich::checking_spike(const double st)
	{
		if (this->v >= this->thre)
		{
			this->v = this->c;
			this->u += this->d;
			this->spike_checking = true;
			this->spiking_time = st;
		}
		else
		{
			this->spike_checking = false;
		}
		return;
	}



	 __device__ void cuda_s_izkevich::exc_synapse_model(const double inter, const double intra, const double dt)
	{
		double temp = -this->gamma_exc * this->exc_synap + intra + inter;
		this->exc_synap += dt * temp;

		return;
	}


	__device__ void cuda_s_izkevich::inh_synapse_model(const double intra, const double inter, const double dt)
	{
		double temp = -this->gamma_inh * this->inh_synap + intra + inter;
		this->inh_synap += dt * temp;

		return;
	}


	*/
