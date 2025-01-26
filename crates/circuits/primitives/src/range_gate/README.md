# Range Gate

This chip gets requests to verify that a number is in the range `[0, MAX)`.
In the trace, there is a counter column and a multiplicity column.
The counter column is generated using constraints (gates), as opposed to the [RangeCheckerChip](../range/README.md) which uses preprocessed trace.
The difference is that this chip does not use any preprocessed trace.

The `RangeCheckerGateAir` constrains that the `counter` column is increasing from 0 to `MAX - 1` and also leaves the `mult` column unconstrained.

```rust
    builder
        .when_first_row()
        .assert_eq(local.counter, AB::Expr::ZERO);
    builder
        .when_transition()
        .assert_eq(local.counter + AB::Expr::ONE, next.counter);
    builder.when_last_row().assert_eq(
        local.counter,
        AB::F::from_canonical_u32(self.bus.range_max - 1),
    );
```

Other than that, the interaction constraints work similarly to the [RangeCheckerChip](../range/README.md).
