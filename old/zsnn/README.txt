
1.  NVCC 설치 

sudo apt install nvidia-cuda-toolkit


2.  boost 설치 

sudo apt-get install libboost-all-dev


3. Python 개발 헤더 파일 설치

sudo apt-get install python3-dev
pip install networkx
pip install numpy 
pip install matplot

4. Boost.Python과 CUDA 코드를 컴파일하여 라이브러리 생성

nvcc -Xcompiler -fpic -arch=sm_75 -O3 -I /usr/include/python3.8 -I /usr/include/boost -shared -o MONET_SNN_CUDA_PYTHON.so  CUDA_DM_NEURON.cu CUDA_DM_RUN.cu boost_python_snn.cpp -lboost_python38 -lpython3.8


5. python 실행 

python3 ./zebrafish_model.py
