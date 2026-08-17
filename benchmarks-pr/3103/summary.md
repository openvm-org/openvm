| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-15adf433f75c61127586f05b670680b73bdc550d.md) | 466 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-15adf433f75c61127586f05b670680b73bdc550d.md) | 7,560 |  14,365,133 |  1,546 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-15adf433f75c61127586f05b670680b73bdc550d.md) | 4,163 |  11,167,961 |  518 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-15adf433f75c61127586f05b670680b73bdc550d.md) | 654 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-15adf433f75c61127586f05b670680b73bdc550d.md) | 197 |  112,210 |  198 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-15adf433f75c61127586f05b670680b73bdc550d.md) | 240 |  592,827 |  204 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-15adf433f75c61127586f05b670680b73bdc550d.md) | 2,030 |  1,979,971 |  530 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/15adf433f75c61127586f05b670680b73bdc550d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32049941999)
