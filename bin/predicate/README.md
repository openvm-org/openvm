# AFS Predicate Binary

## Instructions

Display help:

```bash
cargo run --bin predicate -- --help
```

## Commands

Run these commands from the root of the repository.

### Keygen

Generate proving and verifying keys and save them to disk

```bash
cargo run --release --bin predicate -- keygen -p lt
```

### Prove

Generate a proof of the predicate operation on the table

```bash
cargo run --release --bin predicate -- prove -p lt -v 0x20 -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 -d bin/common/data/input_file_32_32.mockdb -i bin/common/data/predicate/0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711.cache.bin
```

### Verify

Verify the generated proof

```bash
cargo run --release --bin predicate -- verify -p lt -v 0x20 -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711
```

## Full test

Run from the root of the repository.

```bash
# Relevant lines from `config.toml`
[page]
index_bytes = 32
data_bytes = 32
bits_per_fe = 16
height = 256
```

```bash
# Write test input file to mockdb
cargo run --release --bin afs -- mock write -f bin/common/data/test_input_file_32_32.afi -o bin/common/data/input_file_32_32.mockdb

# Cache table input trace
cargo run --release --bin afs -- cache -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 --db-file bin/common/data/input_file_32_32.mockdb --output-folder bin/common/data/predicate

# Generate proving and verifying keys
cargo run --release --bin predicate -- keygen -p lt

# Prove the inputs and save the proof to file
cargo run --release --bin predicate -- prove -p lt -v 0x20 -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 -d bin/common/data/input_file_32_32.mockdb -i bin/common/data/predicate/0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711.cache.bin

# Verify the proof
cargo run --release --bin predicate -- verify -p lt -v 0x20 -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711
```
