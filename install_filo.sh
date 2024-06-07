#!/bin/bash

# variables
current_dir=$(pwd)
vrp_solver_dir="$current_dir/rm/envs/vrp/vrp_solver_filo"

# clone repositories
git clone https://github.com/acco93/cobra.git $vrp_solver_dir/repos/cobra/
git clone https://github.com/acco93/filo.git $vrp_solver_dir/repos/filo/

# modify files
cp $vrp_solver_dir/modified_files/AbstractInstanceParser.hpp $vrp_solver_dir/repos/cobra/include/cobra/AbstractInstanceParser.hpp
cp $vrp_solver_dir/modified_files/main.cpp $vrp_solver_dir/repos/filo/main.cpp

# build cobra
cd $vrp_solver_dir/repos/cobra/
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$vrp_solver_dir/repos/
make -j
make install
export cobra_DIR=$vrp_solver_dir/repos/lib64/cmake/cobra/

# build filo
cd $vrp_solver_dir/repos/filo
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_VERBOSE=OFF
make -j
cp filo $vrp_solver_dir/filo

# back to main dir
cd $current_dir