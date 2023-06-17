
Build Instructions
```
git clone
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/usr/local/libtorch ..
make
```

Test with V8 API 
```
unset TORCH_CUDNN_V8_API_DISABLED
./src/main/
```

Test without V8 API 
```
export TORCH_CUDNN_V8_API_DISABLED=1
./src/main/
```
