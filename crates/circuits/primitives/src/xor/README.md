# XOR Chip

This chip gets requests to compute the xor of two numbers `x` and `y` of at most `M` bits.

It generates a preprocessed table with a row for each possible triple `(x, y, x^y)`
and keeps count of the number of times each triple is requested for the single main trace column.

The `XorLookupAir` adds interaction constraints for each triple `(x, y, x^y)` requested.
```rust
    self.bus
        .receive(prep_local.x, prep_local.y, prep_local.z)
        .eval(builder, local.mult);
```

Then similar to the [RangeCheckerChip](../range/README.md), if the non-materialized send and receive multisets on the shared `XorBus` are equal, then the xor lookup is satisfied.
