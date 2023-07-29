#!/usr/bash

export numThreads=10
# rm -rf build
rm disp
mkdir -p build
cd build
# cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DCMAKE_BUILD_TYPE=Debug ..
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
cp src/disp ..
cd ..
# rm compile_commands.json
# ln -s build/compile_commands.json .
