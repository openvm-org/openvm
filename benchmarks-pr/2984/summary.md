| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/fibonacci-18be4db33696221972a481cbf4dfce1b87bfb7b5.md) | 475 |  4,000,051 |  242 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/keccak-18be4db33696221972a481cbf4dfce1b87bfb7b5.md) | 7,297 |  14,365,133 |  1,535 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/sha2_bench-18be4db33696221972a481cbf4dfce1b87bfb7b5.md) | 4,765 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/regex-18be4db33696221972a481cbf4dfce1b87bfb7b5.md) | 665 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/ecrecover-18be4db33696221972a481cbf4dfce1b87bfb7b5.md) | 222 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/pairing-18be4db33696221972a481cbf4dfce1b87bfb7b5.md) | 266 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/kitchen_sink-18be4db33696221972a481cbf4dfce1b87bfb7b5.md) | 2,692 |  1,979,971 |  470 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/18be4db33696221972a481cbf4dfce1b87bfb7b5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30133262121)
