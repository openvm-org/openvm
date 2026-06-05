| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2843/fibonacci-354b908c73a0e4c1459463c40afbd39a43a7cc59.md) | 3,752 |  12,000,265 |  934 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2843/keccak-354b908c73a0e4c1459463c40afbd39a43a7cc59.md) | 17,821 |  18,655,329 |  3,240 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2843/sha2_bench-354b908c73a0e4c1459463c40afbd39a43a7cc59.md) | 10,039 |  14,793,960 |  1,464 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2843/regex-354b908c73a0e4c1459463c40afbd39a43a7cc59.md) | 1,380 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2843/ecrecover-354b908c73a0e4c1459463c40afbd39a43a7cc59.md) | 600 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2843/pairing-354b908c73a0e4c1459463c40afbd39a43a7cc59.md) | 885 |  1,745,757 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2843/kitchen_sink-354b908c73a0e4c1459463c40afbd39a43a7cc59.md) | 3,822 |  2,579,903 |  943 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/354b908c73a0e4c1459463c40afbd39a43a7cc59

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27017760753)
