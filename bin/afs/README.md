# AFS Query Binary

## Instructions

Display help:

```bash
cargo run --bin afs -- --help
```

## Mock commands

Run these commands from the root of the repository.

### Read

Read from a local mock database file. Set the --db-file (-d), --table-id (-t), and print to stdout with the --print (-p) flag.

```bash
cargo run --bin afs -- mock read -d bin/afs/tests/afs_db.mockdb -t 5
```

### Write

Write to a local mock database file using an AFS Instruction file. Set the --afi-file (-f), --db-file (-d) to write the AFI file into the mock database. Optionally set --print (-p) to print to stdout and --output-db-file (-o) to save the new mock database to file.

```bash
cargo run --bin afs -- mock write -f bin/afs/tests/test_input_file_32_1024.afi -d bin/afs/tests/afs_db.mockdb -o bin/afs/tests/afs_db1.mockdb
```

### AFI

Print the afs instruction set to file.

```bash
cargo run --bin afs -- mock afi -f bin/afs/tests/test_input_file_32_1024.afi
```

## Full test

```bash
cargo run --bin afs -- mock write -f bin/afs/tests/test_input_file_32_1024.afi -o bin/afs/tests/input_file_32_1024.mockdb

cargo run --bin afs -- keygen -o bin/afs/tests       

cargo run --bin afs -- cache -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 -d bin/afs/tests/input_file_32_1024.mockdb -o bin/afs/tests

cargo run --bin afs -- prove -f bin/afs/tests/test_input_file_32_1024.afi -d bin/afs/tests/input_file_32_1024.mockdb -c bin/afs/tests -k bin/afs/tests

cargo run --bin afs -- verify -f bin/afs/tests/input_file_32_1024.mockdb.prove.bin -d '/Users/yujiang/axiom/afs-prototype/bin/afs/tests/input_file_32_1024.mockdb' -k bin/afs/tests
```
