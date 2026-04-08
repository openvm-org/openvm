| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/fibonacci-7e6c07bcc83ce878b9ee9ebd84848375a58d9836.md) | 3,823 |  12,000,265 |  957 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/keccak-7e6c07bcc83ce878b9ee9ebd84848375a58d9836.md) | 18,789 |  18,655,329 |  3,368 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/regex-7e6c07bcc83ce878b9ee9ebd84848375a58d9836.md) | 1,420 |  4,137,067 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/ecrecover-7e6c07bcc83ce878b9ee9ebd84848375a58d9836.md) | 732 |  317,792 |  353 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/pairing-7e6c07bcc83ce878b9ee9ebd84848375a58d9836.md) | 925 |  1,745,757 |  317 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/kitchen_sink-7e6c07bcc83ce878b9ee9ebd84848375a58d9836.md) | 2,369 |  2,580,026 |  785 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7e6c07bcc83ce878b9ee9ebd84848375a58d9836

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24161483252)
