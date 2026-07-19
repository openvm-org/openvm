| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/fibonacci-d1ff49659d406d279eb778f30450d92668035533.md) | 406 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/keccak-d1ff49659d406d279eb778f30450d92668035533.md) | 8,655 |  14,365,133 |  1,531 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/sha2_bench-d1ff49659d406d279eb778f30450d92668035533.md) | 4,202 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/regex-d1ff49659d406d279eb778f30450d92668035533.md) | 569 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/ecrecover-d1ff49659d406d279eb778f30450d92668035533.md) | 218 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/pairing-d1ff49659d406d279eb778f30450d92668035533.md) | 283 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/kitchen_sink-d1ff49659d406d279eb778f30450d92668035533.md) | 1,925 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d1ff49659d406d279eb778f30450d92668035533

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29695846896)
