#/bin/bash
/graphics/opt/opt_Ubuntu18.04/cuda/toolkit_10.0/cuda/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.2
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /home/heid/.local/lib/python2.7/site-packages/tensorflow/include -I /graphics/opt/opt_Ubuntu18.04/cuda/toolkit_10.0/cuda/include -I /home/heid/.local/lib/python2.7/site-packages/tensorflow/include/external/nsync/public -lcudart -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -L /graphics/opt/opt_Ubuntu18.04/cuda/toolkit_10.0/cuda/lib64/ -L/home/heid/.local/lib/python2.7/site-packages/tensorflow -ltensorflow_framework -O2