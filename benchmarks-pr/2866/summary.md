| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/fibonacci-9e8b17ef3c5ed4bb7038931ea78a8add8712ebe3.md) | 3,969 |  12,000,265 |  1,155 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/keccak-9e8b17ef3c5ed4bb7038931ea78a8add8712ebe3.md) | 21,705 |  18,655,329 |  4,593 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/sha2_bench-9e8b17ef3c5ed4bb7038931ea78a8add8712ebe3.md) | 9,677 |  14,793,960 |  1,856 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/regex-9e8b17ef3c5ed4bb7038931ea78a8add8712ebe3.md) | 1,490 |  4,137,067 |  421 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/ecrecover-9e8b17ef3c5ed4bb7038931ea78a8add8712ebe3.md) | 604 |  123,583 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/pairing-9e8b17ef3c5ed4bb7038931ea78a8add8712ebe3.md) | 938 |  1,745,757 |  303 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/kitchen_sink-9e8b17ef3c5ed4bb7038931ea78a8add8712ebe3.md) | 4,097 |  2,579,903 |  873 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9e8b17ef3c5ed4bb7038931ea78a8add8712ebe3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27284965335)
