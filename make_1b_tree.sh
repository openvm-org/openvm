#!/bin/bash
clang++ -Wall -std=c++11 write_file.cc -o write_file -O3
for (( i = 0; i < 1000; i++ ))
do
    echo "$i" 
    ./write_file $i
    echo "FINISHED WRITES"
    cargo run --release --bin afs-1b -- mock write -f tmp.afi -d bin/afs-1b/tests/data/db -o big_tree
done