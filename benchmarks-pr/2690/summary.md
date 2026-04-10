| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2690/fibonacci-95ea2c9ac2430b83625832eab03c639ec554af33.md) | 3,814 |  12,000,265 |  948 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2690/keccak-95ea2c9ac2430b83625832eab03c639ec554af33.md) | 15,655 |  1,235,218 |  2,199 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2690/regex-95ea2c9ac2430b83625832eab03c639ec554af33.md) | 1,403 |  4,136,694 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2690/ecrecover-95ea2c9ac2430b83625832eab03c639ec554af33.md) | 636 |  122,348 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2690/pairing-95ea2c9ac2430b83625832eab03c639ec554af33.md) | 919 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2690/kitchen_sink-95ea2c9ac2430b83625832eab03c639ec554af33.md) | 2,388 |  154,763 |  419 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/95ea2c9ac2430b83625832eab03c639ec554af33

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24255336640)
