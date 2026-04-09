| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/fibonacci-e0dcffc58b03d19dd599c479251cbc27ca7da9d1.md) | 3,819 |  12,000,265 |  944 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/keccak-e0dcffc58b03d19dd599c479251cbc27ca7da9d1.md) | 18,674 |  18,655,329 |  3,353 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/regex-e0dcffc58b03d19dd599c479251cbc27ca7da9d1.md) | 1,434 |  4,137,067 |  379 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/ecrecover-e0dcffc58b03d19dd599c479251cbc27ca7da9d1.md) | 645 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/pairing-e0dcffc58b03d19dd599c479251cbc27ca7da9d1.md) | 910 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/kitchen_sink-e0dcffc58b03d19dd599c479251cbc27ca7da9d1.md) | 2,146 |  2,579,903 |  435 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e0dcffc58b03d19dd599c479251cbc27ca7da9d1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24183012906)
