| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2701/fibonacci-80af5cbd05a17581bdeefaeb5dcc4b465a92abe5.md) | 3,809 |  12,000,265 |  943 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2701/keccak-80af5cbd05a17581bdeefaeb5dcc4b465a92abe5.md) | 18,676 |  18,655,329 |  3,338 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2701/sha2_bench-80af5cbd05a17581bdeefaeb5dcc4b465a92abe5.md) | 9,763 |  14,793,960 |  1,385 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2701/regex-80af5cbd05a17581bdeefaeb5dcc4b465a92abe5.md) | 1,424 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2701/ecrecover-80af5cbd05a17581bdeefaeb5dcc4b465a92abe5.md) | 644 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2701/pairing-80af5cbd05a17581bdeefaeb5dcc4b465a92abe5.md) | 904 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2701/kitchen_sink-80af5cbd05a17581bdeefaeb5dcc4b465a92abe5.md) | 2,152 |  2,579,903 |  435 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/80af5cbd05a17581bdeefaeb5dcc4b465a92abe5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24356123134)
