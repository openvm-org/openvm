| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/fibonacci-63a37c7973ad8a0741e95db82281a73a51996caf.md) | 3,862 |  12,000,265 |  955 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/keccak-63a37c7973ad8a0741e95db82281a73a51996caf.md) | 18,660 |  18,655,329 |  3,331 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/sha2_bench-63a37c7973ad8a0741e95db82281a73a51996caf.md) | 9,822 |  14,793,960 |  1,396 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/regex-63a37c7973ad8a0741e95db82281a73a51996caf.md) | 1,416 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/ecrecover-63a37c7973ad8a0741e95db82281a73a51996caf.md) | 644 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/pairing-63a37c7973ad8a0741e95db82281a73a51996caf.md) | 909 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/kitchen_sink-63a37c7973ad8a0741e95db82281a73a51996caf.md) | 2,158 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/63a37c7973ad8a0741e95db82281a73a51996caf

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24266323873)
