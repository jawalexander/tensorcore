mkdir build
cd build
cmake ..
make VERBOSE=1  -j20
./TensorCoreGEMM
