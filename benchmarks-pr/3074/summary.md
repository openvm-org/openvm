| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/fibonacci-265435a1daa6291ffb7b69954e6a49848d35bee3.md) | 479 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/keccak-265435a1daa6291ffb7b69954e6a49848d35bee3.md) | 7,397 |  14,365,133 |  1,525 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/sha2_bench-265435a1daa6291ffb7b69954e6a49848d35bee3.md) | 4,146 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/regex-265435a1daa6291ffb7b69954e6a49848d35bee3.md) | 662 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/ecrecover-265435a1daa6291ffb7b69954e6a49848d35bee3.md) | 234 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/pairing-265435a1daa6291ffb7b69954e6a49848d35bee3.md) | 239 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/kitchen_sink-265435a1daa6291ffb7b69954e6a49848d35bee3.md) | 2,036 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/265435a1daa6291ffb7b69954e6a49848d35bee3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30502061841)
