#!/bin/bash
rustc input-gen-bin/write_file.rs -o input-gen-bin/write_file -C opt-level=3
rustc input-gen-bin/gen_config-1b.rs -o input-gen-bin/gen_config-1b -C opt-level=3

# Declare the array of tuples
# heights=("1048576 1024" "524288 2048" "262144 4096" "131072 8192" "65536 16384" "32768 32678" "32 32")

heights=("1048576 1024" "262144 4096" "32768 32678" "32 32")

# Loop through each tuple
for tuple in "${heights[@]}"
do
    # Read the values into variables
    echo "$tuple" >> log.txt
    read -r leaf_height internal_height <<< "$tuple"
    ./input-gen-bin/gen_config-1b $leaf_height $internal_height $
    for (( i = 0; i < 100; i++ ))
    do
        echo "$i" 
        ./write_file $i 0
        echo "FINISHED WRITES"
        cargo run --release --bin afs-1b -- mock write -f tmp.afi -d bin/afs-1b/tests/data/db -o big_tree
    done
    ./write_file 2 1
    cargo run --release --bin afs-1b -- keygen -k bin/afs-1b/tests/data/keys >> log.txt
    cargo run --release --bin afs-1b -- prove -f prove_input.afi -d bin/afs-1b/tests/data/db -k bin/afs-1b/tests/data/keys >> log.txt
    cargo run --release --bin afs-1b -- verify -d bin/afs-1b/tests/data/db -k bin/afs-1b/tests/data/keys >> log.txt
done