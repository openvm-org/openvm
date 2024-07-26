#!/bin/bash
rustc input-gen-bin/write_file.rs -o input-gen-bin/write_file -C opt-level=3
rustc input-gen-bin/gen_config-1b.rs -o input-gen-bin/gen_config-1b -C opt-level=3

# Declare the array of tuples
# heights=("1048576 1024" "524288 2048" "262144 4096" "131072 8192" "65536 16384" "32768 32678" "32 32")

# heights=("1048576 1024 2 2" "262144 4096 4 4" "32768 32768 4 4" "32 32 2500 200")
heights=("1048576 1024 2 2")
mkdir input-gen-bin/data
# Loop through each tuple
for tuple in "${heights[@]}"
do
    rm -r bin/afs-1b/tests/data/db
    rm -r bin/afs-1b/tests/data/keys
    # Read the values into variables
    echo "$tuple" >> input-gen-bin/data/log.txt
    echo "$tuple" >> input-gen-bin/data/disk.txt
    read -r leaf_height internal_height leaf_cap internal_cap <<< "$tuple"
    num_ops=65536
    ./input-gen-bin/gen_config-1b $leaf_height $internal_height $leaf_cap $internal_cap $num_ops
    for (( i = 0; i < 1000; i++ ))
    do
        echo "$i" 
        ./input-gen-bin/write_file $i 0
        echo "FINISHED WRITES"
        cargo run --release --bin afs-1b -- mock write -f tmp.afi -d bin/afs-1b/tests/data/db -o big_tree -c >> input-gen-bin/data/log.txt
        du -h bin/afs-1b/tests/data/db >> input-gen-bin/data/disk.txt
    done
    ./input-gen-bin/write_file 2 1
    cargo run --release --bin afs-1b -- keygen -o bin/afs-1b/tests/data/keys >> input-gen-bin/data/log.txt
    cargo run --release --bin afs-1b -- prove -f proof_input.afi -d bin/afs-1b/tests/data/db -k bin/afs-1b/tests/data/keys >> input-gen-bin/data/log.txt
    cargo run --release --bin afs-1b -- verify -d bin/afs-1b/tests/data/db -k bin/afs-1b/tests/data/keys -t big_tree >> input-gen-bin/data/log.txt
done
rm -r bin/afs-1b/tests/data/db
rm -r bin/afs-1b/tests/data/keys
rm tmp.afi
rm proof_input.afi