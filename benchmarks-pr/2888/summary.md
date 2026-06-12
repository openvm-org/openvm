| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/fibonacci-cab59c93358dcbe7a866f097d872e3bb60dcc969.md) | 3,982 |  12,000,265 |  1,146 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/keccak-cab59c93358dcbe7a866f097d872e3bb60dcc969.md) | 21,753 |  18,655,329 |  4,618 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/sha2_bench-cab59c93358dcbe7a866f097d872e3bb60dcc969.md) | 9,684 |  14,793,960 |  1,848 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/regex-cab59c93358dcbe7a866f097d872e3bb60dcc969.md) | 1,497 |  4,137,067 |  425 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/ecrecover-cab59c93358dcbe7a866f097d872e3bb60dcc969.md) | 602 |  123,583 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/pairing-cab59c93358dcbe7a866f097d872e3bb60dcc969.md) | 942 |  1,745,757 |  301 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/kitchen_sink-cab59c93358dcbe7a866f097d872e3bb60dcc969.md) | 4,119 |  2,579,903 |  879 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cab59c93358dcbe7a866f097d872e3bb60dcc969

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27438793264)
