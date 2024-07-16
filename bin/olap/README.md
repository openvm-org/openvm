# OLAP Binary

Perform some OLAP Read operations on a single Page

## Instructions

### Index scan

```bash
WIP
```

### Inner Join

```bash
# Keygen
cargo run --bin olap -- keygen -d bin/olap/tests/data/db.mockdb -f bin/olap/tests/data/innerjoin_0x11_0x12.afo

# Cache
cargo run --bin olap -- cache -d bin/olap/tests/data/db.mockdb -f bin/olap/tests/data/innerjoin_0x11_0x12.afo

# Prove
cargo run --bin olap -- prove -d bin/olap/tests/data/db.mockdb -f bin/olap/tests/data/innerjoin_0x11_0x12.afo

# Verify
cargo run --bin olap -- verify -p bin/olap/tmp/proof.bin

```

### Group By

```bash
cargo run --bin olap -- run -d bin/olap/tests/data/db.mockdb -f bin/olap/tests/data/groupby_0x11.afo
```
