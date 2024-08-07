# AFS Query Binary

## Instructions

Display help:

```bash
cargo run --release --bin afs-1b -- --help
```

## Configuration

All configuration is handled via the `config-1b.toml` file at the root of the repository.

## Commands

Commands to generate keys, cache traces, prove, and verify. Run these commands from the root of the repository.

### keygen

Generate partial proving and verifying keys.

```bash
cargo run --release --bin afs-1b -- keygen --output-folder bin/common/data/keys
```

### prove

Prove a set of instructions.

```bash
cargo run --release --bin afs-1b -- prove --afi-file bin/common/data/test_input_file_32_32_mtrw.afi --db-folder bin/common/data/db --keys-folder bin/common/data/keys
```

### verify

Verify the proof

```bash
cargo run --release --bin afs-1b -- verify --table-id big_tree --db-folder bin/common/data/db --keys-folder bin/common/data/keys
```

## Mock commands

Useful for reading/writing the .mockdb files. Run these commands from the root of the repository.

### Read

Read from a local mock database file. Set the --db-file (-d), --table-id (-t), and print to stdout with the --print (-p) flag.

```bash
cargo run --release --bin afs-1b -- mock read --table-id big_tree --db-folder bin/common/data/db --index 19000050
```

### Write

Write to a local mock database file using an AFS Instruction file. Set the --afi-file (-f), --db-file (-d) to write the AFI file into the mock database. Optionally set --print (-p) to print to stdout and --output-db-file (-o) to save the new mock database to file.

```bash
cargo run --release --bin afs-1b -- mock write -f bin/afs/tests/data/test_input_file_32_32_mtrw.afi -d bin/common/data/db -k bin/common/data/keys
```

### AFI

Print the afs instruction set to file.

```bash
cargo run --release --bin afs-1b -- mock afi -f bin/common/data/test_input_file_32_32_mtrw.afi
```

## Full test

```bash
cargo run --release --bin afs-1b -- keygen --output-folder bin/common/data/keys

cargo run --release --bin afs-1b -- prove --afi-file bin/common/data/test_input_file_32_32_mtrw.afi --db-folder bin/common/data/db --keys-folder bin/common/data/keys

cargo run --release --bin afs-1b -- verify --table-id big_tree --db-folder bin/common/data/db --keys-folder bin/common/data/keys
```
