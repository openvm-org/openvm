| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/fibonacci-a842b24d0073bc5287a1b208c6076a636cd225f1.md) | 3,920 |  12,000,265 |  973 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/keccak-a842b24d0073bc5287a1b208c6076a636cd225f1.md) | 18,771 |  18,655,329 |  3,357 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/sha2_bench-a842b24d0073bc5287a1b208c6076a636cd225f1.md) | 9,104 |  14,793,960 |  1,427 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/regex-a842b24d0073bc5287a1b208c6076a636cd225f1.md) | 1,426 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/ecrecover-a842b24d0073bc5287a1b208c6076a636cd225f1.md) | 652 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/pairing-a842b24d0073bc5287a1b208c6076a636cd225f1.md) | 911 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/kitchen_sink-a842b24d0073bc5287a1b208c6076a636cd225f1.md) | 2,109 |  2,579,903 |  439 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a842b24d0073bc5287a1b208c6076a636cd225f1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24532888964)
