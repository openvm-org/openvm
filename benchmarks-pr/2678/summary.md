| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/fibonacci-8fba73a1189092f77bd15c328463e128141e6edf.md) | 3,842 |  12,000,265 |  955 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/keccak-8fba73a1189092f77bd15c328463e128141e6edf.md) | 18,453 |  18,655,329 |  3,307 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/regex-8fba73a1189092f77bd15c328463e128141e6edf.md) | 1,411 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/ecrecover-8fba73a1189092f77bd15c328463e128141e6edf.md) | 647 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/pairing-8fba73a1189092f77bd15c328463e128141e6edf.md) | 906 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/kitchen_sink-8fba73a1189092f77bd15c328463e128141e6edf.md) | 2,146 |  2,579,903 |  435 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8fba73a1189092f77bd15c328463e128141e6edf

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24257953210)
