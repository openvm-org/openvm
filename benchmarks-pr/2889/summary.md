| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/fibonacci-54f0f48a2ed6775aae537b689870c331bc1dbb42.md) | 3,819 |  12,000,265 |  931 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/keccak-54f0f48a2ed6775aae537b689870c331bc1dbb42.md) | 18,030 |  18,655,329 |  3,281 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/sha2_bench-54f0f48a2ed6775aae537b689870c331bc1dbb42.md) | 10,049 |  14,793,960 |  1,466 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/regex-54f0f48a2ed6775aae537b689870c331bc1dbb42.md) | 1,417 |  4,137,067 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/ecrecover-54f0f48a2ed6775aae537b689870c331bc1dbb42.md) | 603 |  123,583 |  253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/pairing-54f0f48a2ed6775aae537b689870c331bc1dbb42.md) | 889 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/kitchen_sink-54f0f48a2ed6775aae537b689870c331bc1dbb42.md) | 1,871 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/54f0f48a2ed6775aae537b689870c331bc1dbb42

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27441004084)
