## AIRs
1. RangeCheckerAir
2. PowerCheckerAir
3. ExpBitsLenAir

## Components relationship

```text
Graph 1: RangeChecker AIR

                     add_key_with_lookups [RangeCheckerBus]
+------------------+ ----------------------------------------> +----------------------------------+
| RangeChecker AIR |                                           | lookup AIRs on RangeCheckerBus  |
+------------------+                                           | - ProofShapeAir                 |
                                                               +----------------------------------+


Graph 2: PowerChecker AIR

                     add_key_with_lookups [PowerCheckerBus]
+------------------+ ----------------------------------------> +----------------------------------+
| PowerChecker AIR |                                           | lookup AIRs on PowerCheckerBus  |
+------------------+                                           | - ProofShapeAir                 |
                                                               | - ExpressionClaimAir            |
                                                               +----------------------------------+

                     add_key_with_lookups [RangeCheckerBus]
+------------------+ ----------------------------------------> +----------------------------------+
| PowerChecker AIR |                                           | lookup AIRs on RangeCheckerBus  |
+------------------+                                           | - ProofShapeAir                 |
                                                               +----------------------------------+


Graph 3: ExpBitsLen AIR

                     add_key_with_lookups [ExpBitsLenBus]
+------------------+ ----------------------------------------> +----------------------------------+
| ExpBitsLen AIR   |                                           | lookup AIRs on ExpBitsLenBus    |
+------------------+                                           | - GkrInputAir                   |
                                                               | - StackingClaimsAir             |
                                                               | - SumcheckAir                   |
                                                               | - WhirRoundAir                  |
                                                               | - WhirQueryAir                  |
                                                               +----------------------------------+
```


## Notes:

1. Both PowerChecker AIR and RangeChecker AIR do `add_key_with_lookups` on `RangeCheckerBus`, but they are currently for different ranges:
PowerChecker has `max_bits = 5` and RangeChecker has `max_bits = 8`.
