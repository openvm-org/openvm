| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/fibonacci-dc8f178291c11e091c2b67ec04fbe0c89eb4c4de.md) | 1,859 |  4,000,051 |  445 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/keccak-dc8f178291c11e091c2b67ec04fbe0c89eb4c4de.md) | 14,162 |  14,365,133 |  2,398 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/sha2_bench-dc8f178291c11e091c2b67ec04fbe0c89eb4c4de.md) | 8,317 |  11,167,961 |  914 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/regex-dc8f178291c11e091c2b67ec04fbe0c89eb4c4de.md) | 1,565 |  4,090,656 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/ecrecover-dc8f178291c11e091c2b67ec04fbe0c89eb4c4de.md) | 605 |  112,210 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/pairing-dc8f178291c11e091c2b67ec04fbe0c89eb4c4de.md) | 738 |  592,827 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/kitchen_sink-dc8f178291c11e091c2b67ec04fbe0c89eb4c4de.md) | 1,889 |  1,979,971 |  408 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dc8f178291c11e091c2b67ec04fbe0c89eb4c4de

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25936317150)
