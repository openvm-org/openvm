| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/fibonacci-64c2eea14ec3cca92bf8756709e684158b1fb039.md) | 3,824 |  12,000,265 |  952 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/keccak-64c2eea14ec3cca92bf8756709e684158b1fb039.md) | 18,502 |  18,655,329 |  3,303 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/regex-64c2eea14ec3cca92bf8756709e684158b1fb039.md) | 1,426 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/ecrecover-64c2eea14ec3cca92bf8756709e684158b1fb039.md) | 643 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/pairing-64c2eea14ec3cca92bf8756709e684158b1fb039.md) | 903 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/kitchen_sink-64c2eea14ec3cca92bf8756709e684158b1fb039.md) | 2,149 |  2,579,903 |  432 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/64c2eea14ec3cca92bf8756709e684158b1fb039

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24186548999)
