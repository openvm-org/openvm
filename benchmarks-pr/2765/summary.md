| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-f8732f62e83d7ca775b6e09605b8ad1c470d56fb.md) | 1,879 |  4,000,051 |  512 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-f8732f62e83d7ca775b6e09605b8ad1c470d56fb.md) | 13,317 |  14,365,133 |  2,173 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-f8732f62e83d7ca775b6e09605b8ad1c470d56fb.md) | 9,466 |  11,167,961 |  1,409 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-f8732f62e83d7ca775b6e09605b8ad1c470d56fb.md) | 1,564 |  4,090,656 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-f8732f62e83d7ca775b6e09605b8ad1c470d56fb.md) | 594 |  112,210 |  263 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-f8732f62e83d7ca775b6e09605b8ad1c470d56fb.md) | 736 |  592,827 |  257 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-f8732f62e83d7ca775b6e09605b8ad1c470d56fb.md) | 1,873 |  1,979,971 |  410 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f8732f62e83d7ca775b6e09605b8ad1c470d56fb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26252950437)
