| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/fibonacci-b1de4ba8795c05170c73ada0525fc66bbdcc30a0.md) | 474 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/keccak-b1de4ba8795c05170c73ada0525fc66bbdcc30a0.md) | 7,808 |  14,365,133 |  1,645 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/sha2_bench-b1de4ba8795c05170c73ada0525fc66bbdcc30a0.md) | 4,375 |  11,167,961 |  535 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/regex-b1de4ba8795c05170c73ada0525fc66bbdcc30a0.md) | 763 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/ecrecover-b1de4ba8795c05170c73ada0525fc66bbdcc30a0.md) | 206 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/pairing-b1de4ba8795c05170c73ada0525fc66bbdcc30a0.md) | 250 |  592,827 |  172 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3150/kitchen_sink-b1de4ba8795c05170c73ada0525fc66bbdcc30a0.md) | 2,241 |  1,979,971 |  472 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b1de4ba8795c05170c73ada0525fc66bbdcc30a0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33812947501)
