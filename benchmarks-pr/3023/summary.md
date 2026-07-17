| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/fibonacci-a9ced01e9ffcd914c3a9d3258d8e37d521e5be19.md) | 410 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/keccak-a9ced01e9ffcd914c3a9d3258d8e37d521e5be19.md) | 8,587 |  14,365,133 |  1,523 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/sha2_bench-a9ced01e9ffcd914c3a9d3258d8e37d521e5be19.md) | 4,168 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/regex-a9ced01e9ffcd914c3a9d3258d8e37d521e5be19.md) | 577 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/ecrecover-a9ced01e9ffcd914c3a9d3258d8e37d521e5be19.md) | 228 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/pairing-a9ced01e9ffcd914c3a9d3258d8e37d521e5be19.md) | 289 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/kitchen_sink-a9ced01e9ffcd914c3a9d3258d8e37d521e5be19.md) | 1,915 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a9ced01e9ffcd914c3a9d3258d8e37d521e5be19

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29563262870)
