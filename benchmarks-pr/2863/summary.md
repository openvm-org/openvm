| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2863/fibonacci-9b49e342f8f812cc7adeccaccdb44bd196f504ff.md) | 3,944 |  12,000,265 |  1,133 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2863/keccak-9b49e342f8f812cc7adeccaccdb44bd196f504ff.md) | 22,069 |  18,655,329 |  4,677 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2863/sha2_bench-9b49e342f8f812cc7adeccaccdb44bd196f504ff.md) | 9,710 |  14,793,960 |  1,853 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2863/regex-9b49e342f8f812cc7adeccaccdb44bd196f504ff.md) | 1,515 |  4,137,067 |  430 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2863/ecrecover-9b49e342f8f812cc7adeccaccdb44bd196f504ff.md) | 602 |  123,583 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2863/pairing-9b49e342f8f812cc7adeccaccdb44bd196f504ff.md) | 934 |  1,745,757 |  303 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2863/kitchen_sink-9b49e342f8f812cc7adeccaccdb44bd196f504ff.md) | 4,138 |  2,579,903 |  880 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9b49e342f8f812cc7adeccaccdb44bd196f504ff

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27235780603)
