| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2839/fibonacci-ecda825de8a2f44e1774381c910af2c1723099bc.md) | 3,753 |  12,000,265 |  912 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2839/keccak-ecda825de8a2f44e1774381c910af2c1723099bc.md) | 17,861 |  18,655,329 |  3,247 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2839/sha2_bench-ecda825de8a2f44e1774381c910af2c1723099bc.md) | 10,021 |  14,793,960 |  1,463 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2839/regex-ecda825de8a2f44e1774381c910af2c1723099bc.md) | 1,407 |  4,137,067 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2839/ecrecover-ecda825de8a2f44e1774381c910af2c1723099bc.md) | 604 |  123,583 |  254 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2839/pairing-ecda825de8a2f44e1774381c910af2c1723099bc.md) | 873 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2839/kitchen_sink-ecda825de8a2f44e1774381c910af2c1723099bc.md) | 3,848 |  2,579,903 |  952 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ecda825de8a2f44e1774381c910af2c1723099bc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26844106306)
