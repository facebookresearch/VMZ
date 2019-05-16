# Install Caffe2, OpenCV3, FFMPEG

* [CentOS](#centos)  
* [Ubuntu 16.04 (Test only on CPU)](#ubuntu)

---

# CentOS
## Install OpenCV 3.4.0

1. Install and create a virtualenv
```
sudo yum install python-virtualenv  
cd ~/local  
virtualenv r21d  
source r21d/bin/activate  
pip install pip setuptools -U  
```

2. cmake  
Run `cmake --version` to see what version you have, If it doesn't exist or less than 3.7, run
```
sudo yum remove cmake3
sudo yum install cmake
```

3. Get OpenCV
```
wget https://github.com/opencv/opencv/archive/3.4.0.zip -O opencv-3.4.0.zip
unzip opencv-3.4.0.zip
cd opencv-3.4.0
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D BUILD_EXAMPLES=ON \
	-D BUILD_SHARED_LIBS=ON ..
make -j8
sudo make install
sudo ldconfig
```
* If you’ve reached this step without an error, OpenCV should now be installed in  
  `/usr/local/lib/python2.7/site-packages`
* To use OpenCV within r21d virtual environment
  ```
  cd ~/local/r21d/lib/python2.7/site-packages/
  ln -s /usr/local/lib/python2.7/site-packages/cv2.so cv2.so
  ```
* To confirm your installation
  (ensure that you are in the r21d virtual environment)
  ``` console
  $ python
  >>> import cv2
  >>> cv2.__version__
  '3.4.0'
  ```
  
## Install FFmpeg

1. Get the dependencies
```
sudo yum install autoconf automake bzip2 freetype-devel gcc gcc-c++ git libtool pkgconfig zlib-devel yasm-devel libtheora-devel libvorbis-devel libX11-devel gtk2-devel
```

2. H.264 video encoder
```
cd ~/local
git clone http://git.videolan.org/git/x264.git
cd x264
./configure --enable-shared --enable-pic 
make -j8
sudo make install
```

3. Ogg bitstream library
```
cd ~/local
curl -O -L http://downloads.xiph.org/releases/ogg/libogg-1.3.3.tar.gz
tar xzvf libogg-1.3.3.tar.gz
cd libogg-1.3.3
./configure
make -j8
sudo make install
```

4. FFmpeg
```
git clone http://git.videolan.org/git/ffmpeg.git
cd ffmpeg
 ./configure --enable-gpl --enable-nonfree --enable-libtheora --enable-libvorbis  --enable-libx264  --enable-postproc --enable-version3 --enable-pthreads --enable-shared --enable-pic
make -j8
sudo make install
```
* Make sure pkg config and the linker can see ffmpeg
   ```
   export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH 
   export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
   hash -r 
   sudo ldconfig
   ```
* To confirm your installation
  ```console
  $ ffmpeg
  ffmpeg version N-91042-ga3a6d4d Copyright (c) 2000-2018 the FFmpeg developers
  built with gcc 4.8.5 (GCC) 20150623 (Red Hat 4.8.5-16)
  configuration: --enable-gpl --enable-nonfree --enable-libtheora --enable-libvorbis --enable-libx264 --enable-postproc --enable-version3 --enable-pthreads --enable-shared --enable-pic
  ```

## Install Caffe2

1. Get the dependencies
```
sudo yum install -y protobuf-devel leveldb-devel snappy-devel opencv-devel lmdb-devel python-devel gflags-devel glog-devel kernel-devel
```

2. Install [cuDNN](https://developer.nvidia.com/cudnn)
```
cp cuda/lib64/* /usr/local/cuda/lib64/
cp cuda/include/cudnn.h /usr/local/cuda/include/
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
 * Make sure nvcc is runnable
   ```console
   $ nvcc --version
   nvcc: NVIDIA (R) Cuda compiler driver
   Copyright (c) 2005-2016 NVIDIA Corporation
   Built on Sun_Sep__4_22:14:01_CDT_2016
   Cuda compilation tools, release 8.0, V8.0.44
   ```

3. Install Python dependencies
```
sudo pip install lmdb numpy flask future graphviz hypothesis jupyter matplotlib protobuf pydot python-nvd3 pyyaml requests scikit-image scipy six tornado
```

4. Build Caffe2
```
cd ~/local
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch && git submodule update --init
```
  * Modify CMakeLists.txt to make `USE_FFMPEG ON`
  * After `cmake` (see below), check the output log, makesure `USE_OPENCV: ON` and `USE_FFMPEG: ON`
```
mkdir build
cd build
cmake ..
sudo make -j8 install
```

```
export PYTHONPATH=$PYTHONPATH:~/local/pytorch/build
```
  
  * To confirm your installation
  ```
  python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
  ```
---
# Ubuntu
## Install OpenCV 3.4.0

1. Update `apt-get` and pre-installed libraries
```
sudo apt-get update
sudo apt-get upgrade
```

2. Install some tools
```
sudo apt-get install build-essential cmake pkg-config libatlas-base-dev gfortran unzip
sudo apt-get install python2.7-dev
sudo apt install python-pip
```

3. Setup virtualenv
```
sudo pip install virtualenv virtualenvwrapper
echo -e "\n# virtualenv and virtualenvwrapper" >> ~/.bashrc
echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
source ~/.bashrc
mkvirtualenv r21d -p python2
pip install numpy 
```
* Note that we need to install numpy before compile OpenCV
* Ensure you are in the correct virtual environment (e.g. `r21d`)
4. Get OpenCV
```
wget https://github.com/opencv/opencv/archive/3.4.0.zip -O opencv-3.4.0.zip
unzip opencv-3.4.0.zip
cd opencv-3.4.0
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D PYTHON_EXECUTABLE=~/.virtualenvs/r21d/bin/python \
        -D BUILD_EXAMPLES=ON \
	-D BUILD_SHARED_LIBS=ON ..
make -j8
sudo make install
sudo ldconfig
```
* If you’ve reached this step without an error, OpenCV should now be installed in  
  `/usr/local/lib/python2.7/site-packages`
* To use OpenCV within r21d virtual environment
  ```
  cd ~/.virtualenvs/r21d/lib/python2.7/site-packages/
  ln -s /usr/local/lib/python2.7/site-packages/cv2.so cv2.so
  ```
* To confirm your installation
  (ensure that you are in the r21d virtual environment)
  ``` console
  $ python
  >>> import cv2
  >>> cv2.__version__
  '3.4.0'
  ```
  
## Install FFmpeg 

1. Get the dependencies
```
sudo apt-get update -qq && sudo apt-get -y install \
  autoconf \
  automake \
  build-essential \
  cmake \
  git-core \
  libass-dev \
  libfreetype6-dev \
  libsdl2-dev \
  libtool \
  libva-dev \
  libvdpau-dev \
  libvorbis-dev \
  libxcb1-dev \
  libxcb-shm0-dev \
  libxcb-xfixes0-dev \
  pkg-config \
  texinfo \
  wget \
  zlib1g-dev

sudo apt-get install yasm libx264-dev

```

2. In your home directory make a new directory to put all of the source code into:
```
mkdir -p ~/ffmpeg_sources ~/bin
```

3. NASM
```
cd ~/ffmpeg_sources && \
wget https://www.nasm.us/pub/nasm/releasebuilds/2.13.03/nasm-2.13.03.tar.bz2 && \
tar xjvf nasm-2.13.03.tar.bz2 && \
cd nasm-2.13.03 && \
./autogen.sh && \
PATH="$HOME/bin:$PATH" ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin" && \
make && \
make install
```

4. FFmpeg
```
cd ~/ffmpeg_sources && \
wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
tar xjvf ffmpeg-snapshot.tar.bz2 && \
cd ffmpeg && \
PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
  --prefix="$HOME/ffmpeg_build" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I$HOME/ffmpeg_build/include" \
  --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
  --extra-libs="-lpthread -lm" \
  --bindir="$HOME/bin" \
  --enable-gpl \
  --enable-libvorbis \
  --enable-libx264 \
  --enable-nonfree
PATH="$HOME/bin:$PATH" 
make -j8
make install
```
* Make sure pkg config and the linker can see ffmpeg
   ```
   echo "export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
   echo "export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH" >> ~/.bashrc
   hash -r 
   sudo ldconfig
   ```
* To confirm your installation
  ```console
  $ ffmpeg
  ffmpeg version N-91283-gdaf38d0 Copyright (c) 2000-2018 the FFmpeg developers
  built with gcc 5.4.0 (Ubuntu 5.4.0-6ubuntu1~16.04.9) 20160609
  ```

## Install Caffe2

1. Get the dependencies
```
sudo apt-get install -y \
      libgoogle-glog-dev \
      libgtest-dev \
      libiomp-dev \
      libleveldb-dev \
      liblmdb-dev \
      libopencv-dev \
      libopenmpi-dev \
      libsnappy-dev \
      libprotobuf-dev \
      protobuf-compiler \
      libgflags-dev \
      python-dev

pip install lmdb flask future graphviz hypothesis jupyter matplotlib protobuf pydot python-nvd3 pyyaml requests scikit-image scipy six tornado
```

2. Build Caffe2
* If you have a GPU, consider reference to the [steps](https://caffe2.ai/docs/getting-started.html?platform=ubuntu&configuration=compile#install-with-gpu-support). (note that we didn't test it in this tutorial)
```
cd ~
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch && git submodule update --init

USE_OPENCV=1 USE_FFMPEG=1 USE_LMDB=1 python setup.py install
```

```
export PYTHONPATH=$PYTHONPATH:$HOME/pytorch/build
```
  
  * To confirm your installation
  ```
  python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
  ```
  
