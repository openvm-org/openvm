| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-6a54f3234641dc34f76179e1b546c7d61a64d9bc.md) | 469 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-6a54f3234641dc34f76179e1b546c7d61a64d9bc.md) | 7,174 |  14,365,133 |  1,542 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-6a54f3234641dc34f76179e1b546c7d61a64d9bc.md) | 4,350 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-6a54f3234641dc34f76179e1b546c7d61a64d9bc.md) | 661 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-6a54f3234641dc34f76179e1b546c7d61a64d9bc.md) | 226 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-6a54f3234641dc34f76179e1b546c7d61a64d9bc.md) | 247 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-6a54f3234641dc34f76179e1b546c7d61a64d9bc.md) | 2,721 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6a54f3234641dc34f76179e1b546c7d61a64d9bc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29472212307)
