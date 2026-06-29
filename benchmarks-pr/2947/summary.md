| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/fibonacci-48598786da1c43cde6300323b7a283c121bb277d.md) | 3,022 |  12,000,265 |  665 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/keccak-48598786da1c43cde6300323b7a283c121bb277d.md) | 16,211 |  18,655,329 |  3,012 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/sha2_bench-48598786da1c43cde6300323b7a283c121bb277d.md) | 9,206 |  14,793,960 |  1,123 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/regex-48598786da1c43cde6300323b7a283c121bb277d.md) | 1,163 |  4,137,067 |  349 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/ecrecover-48598786da1c43cde6300323b7a283c121bb277d.md) | 607 |  123,583 |  293 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/pairing-48598786da1c43cde6300323b7a283c121bb277d.md) | 944 |  1,745,757 |  309 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2947/kitchen_sink-48598786da1c43cde6300323b7a283c121bb277d.md) | 4,120 |  2,579,903 |  885 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/48598786da1c43cde6300323b7a283c121bb277d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28405924535)
