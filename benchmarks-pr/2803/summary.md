| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/fibonacci-3eaf625952c75d0851477b99fcf8cd093dc13429.md) | 3,819 |  12,000,265 |  929 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/keccak-3eaf625952c75d0851477b99fcf8cd093dc13429.md) | 19,205 |  18,655,329 |  3,391 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/sha2_bench-3eaf625952c75d0851477b99fcf8cd093dc13429.md) | 10,109 |  14,793,960 |  1,451 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/regex-3eaf625952c75d0851477b99fcf8cd093dc13429.md) | 1,402 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/ecrecover-3eaf625952c75d0851477b99fcf8cd093dc13429.md) | 601 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/pairing-3eaf625952c75d0851477b99fcf8cd093dc13429.md) | 893 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/kitchen_sink-3eaf625952c75d0851477b99fcf8cd093dc13429.md) | 1,898 |  2,579,903 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3eaf625952c75d0851477b99fcf8cd093dc13429

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26306418972)
