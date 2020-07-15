
### Build with Dockerfile

```docker build . -t pytorch:1.5.0```


### Excute 

```docker run -it pytorch:1.5.0 /bin/bash```


### Setup miniconda 3

```~/miniconda.sh -b -p /opt/conda```


```rm ~/miniconda.sh```

```/opt/conda/bin/conda install -y python=3.6 numpy pyyaml scipy ipython mkl mkl-include ninja cython typing```


```/opt/conda/bin/conda install -y -c pytorch magma-cuda100 &&  /opt/conda/bin/conda clean -ya```


### Commit and save to image 


```docker commit [container_id] pytorch:1.4.0-conda```


### Reboot a docker 

```docker run -it -v /home/shixiong/pytorch:/opt/pytorch pytorch:1.5.0 /bin/bash```

```cd /opt/pytorch```




### Build PyTorch

```git submodule update --init --recursive``` (maybe outside the docker)


### Fix a small issue of git submoudle update

git pull
git clone 
git fetech url.git commitid
git checkout commitid


### install gcc 7 in docker 

apt-get install -y software-properties-common
add-apt-repository ppa:ubuntu-toolchain-r/test
apt update
apt install g++-7 -y

### Set up PyTorch development 


#### Clean up installs
```conda uninstall pytorch```
```pip uninstall torch```
```pip uninstall torch```

#### setup develop
export DEBUG=1 REL_WITH_DEB_INFO=1 BUILD_TEST=0 BUILD_CAFFE2_OPS=0

export TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" 
export TORCH_NVCC_FLAGS="-Xfatbin -compress-all" 
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop















