| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/fibonacci-82ba6bfddf1e3104b6fdb1940b44505eb2757b37.md) | 448 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/keccak-82ba6bfddf1e3104b6fdb1940b44505eb2757b37.md) | 7,186 |  14,365,133 |  1,594 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/sha2_bench-82ba6bfddf1e3104b6fdb1940b44505eb2757b37.md) | 4,126 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/regex-82ba6bfddf1e3104b6fdb1940b44505eb2757b37.md) | 732 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/ecrecover-82ba6bfddf1e3104b6fdb1940b44505eb2757b37.md) | 203 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/pairing-82ba6bfddf1e3104b6fdb1940b44505eb2757b37.md) | 247 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/kitchen_sink-82ba6bfddf1e3104b6fdb1940b44505eb2757b37.md) | 2,150 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/82ba6bfddf1e3104b6fdb1940b44505eb2757b37

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31730611847)
