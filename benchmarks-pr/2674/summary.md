| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/fibonacci-7ec8d562f38eadd604dcccf0af176c7a7f8cdee0.md) | 3,813 |  12,000,265 |  947 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/keccak-7ec8d562f38eadd604dcccf0af176c7a7f8cdee0.md) | 18,503 |  18,655,329 |  3,304 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/regex-7ec8d562f38eadd604dcccf0af176c7a7f8cdee0.md) | 1,409 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/ecrecover-7ec8d562f38eadd604dcccf0af176c7a7f8cdee0.md) | 641 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/pairing-7ec8d562f38eadd604dcccf0af176c7a7f8cdee0.md) | 901 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/kitchen_sink-7ec8d562f38eadd604dcccf0af176c7a7f8cdee0.md) | 2,160 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7ec8d562f38eadd604dcccf0af176c7a7f8cdee0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24154707237)
