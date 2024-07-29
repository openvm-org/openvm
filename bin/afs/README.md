# AFS Query Binary

## Instructions

Display help:

```bash
cargo run --release --bin afs -- --help
```

## Configuration

All configuration is handled via the `config.toml` file at the root of the repository.

## Commands

Commands to generate keys, cache traces, prove, and verify. Run these commands from the root of the repository.

### keygen

Generate partial proving and verifying keys.

```bash
cargo run --release --bin afs -- keygen --output-folder bin/common/data
```

### cache

Cache a trace of a table.

```bash
cargo run --release --bin afs -- cache -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 --db-file bin/common/data/input_file_32_1024.mockdb --output-folder bin/common/data
```

### prove

Prove a set of instructions.

```bash
cargo run --release --bin afs -- prove --afi-file bin/common/data/test_input_file_32_1024.afi --db-file bin/common/data/input_file_32_1024.mockdb --cache-folder bin/common/data --keys-folder bin/common/data
```

### verify

Verify the proof

```bash
cargo run --release --bin afs -- verify --proof-file bin/common/data/input_file_32_1024.mockdb.prove.bin --db-file bin/common/data/input_file_32_1024.mockdb --keys-folder bin/common/data
```

## Mock commands

Useful for reading/writing the .mockdb files. Run these commands from the root of the repository.

### Describe

List all tables and table metadata in a given mock database file. Set the --db-file (-d) flag.

```bash
cargo run --release --bin afs -- mock describe -d bin/common/data/afs_db.mockdb
```

### Read

Read from a local mock database file. Set the --db-file (-d), --table-id (-t), and print to stdout with the --print (-p) flag.

```bash
cargo run --release --bin afs -- mock read -d bin/common/data/afs_db.mockdb -t 5
```

### Write

Write to a local mock database file using an AFS Instruction file. Set the --afi-file (-f), --db-file (-d) to write the AFI file into the mock database. Optionally set --print (-p) to print to stdout and --output-db-file (-o) to save the new mock database to file.

```bash
cargo run --release --bin afs -- mock write -f bin/common/data/test_input_file_32_1024.afi -d bin/common/data/afs_db.mockdb -o bin/common/data/afs_db1.mockdb
```

### AFI

Print the afs instruction set to file.

```bash
cargo run --release --bin afs -- mock afi -f bin/common/data/test_input_file_32_1024.afi
```

## Full test

```bash
cargo run --release --bin afs -- mock write -f bin/common/data/test_input_file_32_1024.afi -o bin/common/data/input_file_32_1024.mockdb

cargo run --release --bin afs -- keygen --output-folder bin/common/data

cargo run --release --bin afs -- cache -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 --db-file bin/common/data/input_file_32_1024.mockdb --output-folder bin/common/data

cargo run --release --bin afs -- prove --afi-file bin/common/data/test_input_file_32_1024.afi --db-file bin/common/data/input_file_32_1024.mockdb --cache-folder bin/common/data --keys-folder bin/common/data

cargo run --release --bin afs -- verify --proof-file bin/common/data/input_file_32_1024.mockdb.prove.bin --db-file bin/common/data/input_file_32_1024.mockdb --keys-folder bin/common/data
```
