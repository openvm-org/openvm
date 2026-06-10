| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci-64bfcb2a7c26d5fec95f439cbfd1dd6e8d0d5b17.md) | 4,026 |  12,000,265 |  1,159 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/keccak-64bfcb2a7c26d5fec95f439cbfd1dd6e8d0d5b17.md) | 21,704 |  18,655,329 |  4,583 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/sha2_bench-64bfcb2a7c26d5fec95f439cbfd1dd6e8d0d5b17.md) | 9,650 |  14,793,960 |  1,861 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex-64bfcb2a7c26d5fec95f439cbfd1dd6e8d0d5b17.md) | 1,499 |  4,137,067 |  425 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover-64bfcb2a7c26d5fec95f439cbfd1dd6e8d0d5b17.md) | 603 |  123,583 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing-64bfcb2a7c26d5fec95f439cbfd1dd6e8d0d5b17.md) | 937 |  1,745,757 |  305 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink-64bfcb2a7c26d5fec95f439cbfd1dd6e8d0d5b17.md) | 4,196 |  2,579,903 |  902 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci_e2e-64bfcb2a7c26d5fec95f439cbfd1dd6e8d0d5b17.md) | 1,708 |  12,000,265 |  491 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex_e2e-64bfcb2a7c26d5fec95f439cbfd1dd6e8d0d5b17.md) | 712 |  4,137,067 |  198 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover_e2e-64bfcb2a7c26d5fec95f439cbfd1dd6e8d0d5b17.md) | 367 |  123,583 |  143 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing_e2e-64bfcb2a7c26d5fec95f439cbfd1dd6e8d0d5b17.md) | 507 |  1,745,757 |  147 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink_e2e-64bfcb2a7c26d5fec95f439cbfd1dd6e8d0d5b17.md) | 2,165 |  2,579,903 |  389 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/64bfcb2a7c26d5fec95f439cbfd1dd6e8d0d5b17

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27291679115)
