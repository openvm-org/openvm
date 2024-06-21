# AFS Predicate Binary

## Instructions

Display help:

```bash
cargo run --bin predicate -- --help
```

## Mock commands

Run these commands from the root of the repository.

### eq (==)

Find all indexes from a tableId (-t) in a mockDb (-d) that are equal to some value (-v).

```bash
cargo run --bin predicate -- eq -d bin/afs/tests/afs_db.mockdb -t 5 -v 100
```

### lt (<)

Find all indexes from a tableId (-t) in a mockDb (-d) that are less than some value (-v).

```bash
cargo run --bin predicate -- lt -d bin/afs/tests/afs_db.mockdb -t 5 -v 100
```

### lte (<=)

Find all indexes from a tableId (-t) in a mockDb (-d) that are less than or equal to some value (-v).

```bash
cargo run --bin predicate -- lte -d bin/afs/tests/afs_db.mockdb -t 5 -v 100
```

## Full test

Run from the root of the repository.

```bash
cargo run --release --bin afs -- mock write -f bin/afs/tests/data/test_input_file_32_32.afi -o bin/afs/tests/data/input_file_32_32.mockdb

cargo run --bin predicate -- eq -d bin/afs/tests/data/input_file_32_32.mockdb -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 -v 19000050
```
