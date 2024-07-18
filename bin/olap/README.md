# OLAP Binary

Perform some OLAP Read operations on a single Page

## Instructions

### Filter

```bash
# Keygen
cargo run --bin olap -- keygen -d bin/olap/tests/data/db.mockdb -f bin/olap/tests/data/filter_0x11.afo

# Cache
cargo run --bin olap -- cache -d bin/olap/tests/data/db.mockdb -f bin/olap/tests/data/filter_0x11.afo

# Prove
cargo run --bin olap -- prove -d bin/olap/tests/data/db.mockdb -f bin/olap/tests/data/filter_0x11.afo

# Verify
cargo run --bin olap -- verify -d bin/olap/tests/data/db.mockdb -f bin/olap/tests/data/filter_0x11.afo
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
cargo run --bin olap -- verify -d bin/olap/tests/data/db.mockdb -f bin/olap/tests/data/innerjoin_0x11_0x12.afo
```

### Group By

```bash
WIP
```
