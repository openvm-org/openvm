# OLAP Binary

Perform some OLAP Read operations on a single Page

## Instructions

### Index scan

```bash
WIP
```

### Inner Join

```bash
cargo run --bin olap -- run -d bin/olap/tests/data/db.mockdb -f bin/olap/tests/data/innerjoin_0x11_0x12.afo
```

### Group By

```bash
cargo run --bin olap -- run -d bin/olap/tests/data/db.mockdb -f bin/olap/tests/data/groupby_0x11.afo
```
