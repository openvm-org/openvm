| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/fibonacci-83e2cd12001b76eac7e0a73768f8542f06c75ecc.md) | 3,804 |  12,000,265 |  950 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/keccak-83e2cd12001b76eac7e0a73768f8542f06c75ecc.md) | 18,497 |  18,655,329 |  3,304 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/regex-83e2cd12001b76eac7e0a73768f8542f06c75ecc.md) | 1,442 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/ecrecover-83e2cd12001b76eac7e0a73768f8542f06c75ecc.md) | 650 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/pairing-83e2cd12001b76eac7e0a73768f8542f06c75ecc.md) | 905 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/kitchen_sink-83e2cd12001b76eac7e0a73768f8542f06c75ecc.md) | 2,286 |  2,579,903 |  444 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/83e2cd12001b76eac7e0a73768f8542f06c75ecc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24101553576)
