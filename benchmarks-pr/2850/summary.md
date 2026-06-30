| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/fibonacci-ff751957a4cfeefa242fb1cf49f325dace218a42.md) | 3,341 |  4,000,051 |  388 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/keccak-ff751957a4cfeefa242fb1cf49f325dace218a42.md) | 20,564 |  14,365,133 |  3,055 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/sha2_bench-ff751957a4cfeefa242fb1cf49f325dace218a42.md) | 10,281 |  11,167,961 |  992 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/regex-ff751957a4cfeefa242fb1cf49f325dace218a42.md) | 2,607 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/ecrecover-ff751957a4cfeefa242fb1cf49f325dace218a42.md) | 1,957 |  112,210 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/pairing-ff751957a4cfeefa242fb1cf49f325dace218a42.md) | 2,109 |  592,827 |  297 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/kitchen_sink-ff751957a4cfeefa242fb1cf49f325dace218a42.md) | 5,609 |  1,979,971 |  867 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ff751957a4cfeefa242fb1cf49f325dace218a42

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28467255250)
