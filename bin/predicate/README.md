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
cargo run --bin predicate -- keygen -p lt -v 0x20 -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 -d bin/common/data/input_file_32_32.mockdb
```

### Prove

Generate a proof of the predicate operation on the table

```bash
cargo run --bin predicate -- prove -p lt -v 0x20 -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 -d bin/common/data/input_file_32_32.mockdb
```

### Verify

Verify the generated proof

```bash
cargo run --bin predicate -- verify -p lt -v 0x20 -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 -d bin/common/data/input_file_32_32.mockdb
```

## Full test

Run from the root of the repository.

```bash
# config.toml
[page]
index_bytes = 32
data_bytes = 32
bits_per_fe = 16
height = 256
```

```bash
cargo run --release --bin afs -- mock write -f bin/common/data/test_input_file_32_32.afi -o bin/common/data/input_file_32_32.mockdb

cargo run --bin predicate -- keygen -p lt

cargo run --bin predicate -- prove -p lt -v 0x20 -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 -d bin/common/data/input_file_32_32.mockdb

cargo run --bin predicate -- verify -p lt -v 0x20 -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 -d bin/common/data/input_file_32_32.mockdb
```
