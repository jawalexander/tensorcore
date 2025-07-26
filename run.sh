
git config user.name "jawalexander"
git config user.email "jawalexander@outlook.com"

git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890


mkdir build
cd build
clear
set -e
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j32 #  VERBOSE=1 
./TensorCoreGEMMV5
echo "==============*******************************************====================="
# ./TensorCoreGEMMV5
