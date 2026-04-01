| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2645/fibonacci-80674150a38fa9162ea71eb436e27a979ec5fa46.md) | 3,818 |  12,000,265 |  937 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2645/keccak-80674150a38fa9162ea71eb436e27a979ec5fa46.md) | 18,589 |  18,655,329 |  3,320 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2645/regex-80674150a38fa9162ea71eb436e27a979ec5fa46.md) | 1,414 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2645/ecrecover-80674150a38fa9162ea71eb436e27a979ec5fa46.md) | 652 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2645/pairing-80674150a38fa9162ea71eb436e27a979ec5fa46.md) | 912 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2645/kitchen_sink-80674150a38fa9162ea71eb436e27a979ec5fa46.md) | 2,272 |  2,579,903 |  439 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/80674150a38fa9162ea71eb436e27a979ec5fa46

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23853930189)
