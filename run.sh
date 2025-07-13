
git config user.name "jawalexander"
git config user.email "jawalexander@outlook.com"

mkdir build
cd build
clear
set -e
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j32 #  VERBOSE=1 
./TensorCoreGEMMV3
echo "==============*******************************************====================="
./TensorCoreGEMMV4
