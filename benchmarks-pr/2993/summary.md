| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-693ead36f64266f8f8430f7aa74ef939f72d0a85.md) | 415 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-693ead36f64266f8f8430f7aa74ef939f72d0a85.md) | 8,669 |  14,365,133 |  1,541 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-693ead36f64266f8f8430f7aa74ef939f72d0a85.md) | 4,176 |  11,167,961 |  514 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-693ead36f64266f8f8430f7aa74ef939f72d0a85.md) | 579 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-693ead36f64266f8f8430f7aa74ef939f72d0a85.md) | 220 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-693ead36f64266f8f8430f7aa74ef939f72d0a85.md) | 283 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-693ead36f64266f8f8430f7aa74ef939f72d0a85.md) | 1,939 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/693ead36f64266f8f8430f7aa74ef939f72d0a85

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29502483801)
