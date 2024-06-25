# OLAP Binary

Perform some OLAP Read operations on a single Page

## Instructions

### Index scan

```bash
cargo run --bin olap -- run -d bin/common/data/ -f bin/olap/tests/data/indexscan_0x11.afo
```

### Inner Join

```bash
cargo run --bin olap -- run -d bin/common/data/ -f bin/olap/tests/data/innerjoin_0x11_0x12.afo
```

### Group By

```bash
cargo run --bin olap -- run -d bin/common/data/ -f bin/olap/tests/data/groupby_0x11.afo
```
