export CUDA=12.6
export PATH=/usr/local/cuda-$CUDA/bin${PATH:+:${PATH}}
export CUDA_PATH=/usr/local/cuda-$CUDA
export CUDA_HOME=/usr/local/cuda-$CUDA
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export NVCC=/usr/local/cuda-$CUDA/bin/nvcc                                
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
