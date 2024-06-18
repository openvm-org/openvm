# AFS Predicate Index Scan

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
