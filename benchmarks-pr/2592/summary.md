| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-88772862abe8770052577ecbe7ec9eb6b00b1a59.md) | 3,831 |  12,000,265 |  938 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-88772862abe8770052577ecbe7ec9eb6b00b1a59.md) | 18,268 |  18,655,329 |  3,268 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-88772862abe8770052577ecbe7ec9eb6b00b1a59.md) | 1,426 |  4,137,067 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-88772862abe8770052577ecbe7ec9eb6b00b1a59.md) | 650 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-88772862abe8770052577ecbe7ec9eb6b00b1a59.md) | 898 |  1,745,757 |  278 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-88772862abe8770052577ecbe7ec9eb6b00b1a59.md) | 2,290 |  2,579,903 |  441 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/88772862abe8770052577ecbe7ec9eb6b00b1a59

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23808777890)
