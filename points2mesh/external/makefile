nvcc = ${OPT_PATH}/cuda/toolkit_10.0/cuda/bin/nvcc
cudalib = ${OPT_PATH}/cuda/toolkit_10.0/cuda/lib64
tensorflow = /home/heid/.local/lib/python2.7/site-packages/tensorflow/include
TF_LIB = /home/heid/.local/lib/python2.7/site-packages/tensorflow

all: tf_approxmatch_so.so tf_approxmatch_g.cu.o tf_nndistance_so.so tf_nndistance_g.cu.o


tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib)  -D_GLIBCXX_USE_CXX11_ABI=0 


tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC  


tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib)  -L$(TF_LIB) -ltensorflow_framework

#	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib)  -D_GLIBCXX_USE_CXX11_ABI=0 -L$(TF_LIB) -ltensorflow_framework
#	g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o tf_nndistance_so.so tf_nndistance.cpp tf_nndistance_g.cu.o -I $(tensorflow) -DGOOGLE_CUDA=1 -fPIC -DGLIBCXX_USE_CXX11_ABI=0 -lcudart -L $(cudalib)


tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -L$(TF_LIB) -ltensorflow_framework

	#$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -L$(TF_LIB) -ltensorflow_framework
	#$(nvcc) -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler  -fPIC -DGLIBCXX_USE_CXX11_ABI=0 -lcudart -L $(cudalib)


clean:
	rm tf_approxmatch_so.so
	rm tf_nndistance_so.so
	rm  *.cu.o 
