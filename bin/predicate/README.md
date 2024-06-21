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
cargo run --bin predicate -- eq -d bin/common/data/input_file_32_32.mockdb -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 -v 19000050 
```

### gt (<)

Find all indexes from a tableId (-t) in a mockDb (-d) that are greater than some value (-v).

```bash
cargo run --bin predicate -- gt -d bin/common/data/input_file_32_32.mockdb -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 -v 19000050 
```

### gte (<=)

Find all indexes from a tableId (-t) in a mockDb (-d) that are greater than or equal to some value (-v).

```bash
cargo run --bin predicate -- gte -d bin/common/data/input_file_32_32.mockdb -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 -v 19000050 
```

### lt (<)

Find all indexes from a tableId (-t) in a mockDb (-d) that are less than some value (-v).

```bash
cargo run --bin predicate -- lt -d bin/common/data/input_file_32_32.mockdb -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 -v 19000050 
```

### lte (<=)

Find all indexes from a tableId (-t) in a mockDb (-d) that are less than or equal to some value (-v).

```bash
cargo run --bin predicate -- lte -d bin/common/data/input_file_32_32.mockdb -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 -v 19000050 
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
mode = "ReadWrite" # options: "ReadOnly", "ReadWrite"
max_rw_ops = 256
```

```bash
cargo run --release --bin afs -- mock write -f bin/common/data/test_input_file_32_32.afi -o bin/common/data/input_file_32_32.mockdb

cargo run --bin predicate -- eq -d bin/common/data/input_file_32_32.mockdb -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 -v 19000050

cargo run --bin predicate -- gt -d bin/common/data/input_file_32_32.mockdb -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 -v 19000050 

cargo run --bin predicate -- gte -d bin/common/data/input_file_32_32.mockdb -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 -v 19000050 

cargo run --bin predicate -- lt -d bin/common/data/input_file_32_32.mockdb -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 -v 19000050 

cargo run --bin predicate -- lte -d bin/common/data/input_file_32_32.mockdb -t 0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711 -v 19000050 
```
