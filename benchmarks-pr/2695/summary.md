| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/fibonacci-c23b0ca8cdcb5b1937243771f2f77f1e605a65c9.md) | 3,824 |  12,000,265 |  945 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/keccak-c23b0ca8cdcb5b1937243771f2f77f1e605a65c9.md) | 18,695 |  18,655,329 |  3,339 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/regex-c23b0ca8cdcb5b1937243771f2f77f1e605a65c9.md) | 1,442 |  4,137,067 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/ecrecover-c23b0ca8cdcb5b1937243771f2f77f1e605a65c9.md) | 647 |  123,583 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/pairing-c23b0ca8cdcb5b1937243771f2f77f1e605a65c9.md) | 908 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/kitchen_sink-c23b0ca8cdcb5b1937243771f2f77f1e605a65c9.md) | 2,091 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c23b0ca8cdcb5b1937243771f2f77f1e605a65c9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24258362354)
