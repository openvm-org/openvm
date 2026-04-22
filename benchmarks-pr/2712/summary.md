| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/fibonacci-f1bce711a32291920e5cc4cb28b56b371d80f98d.md) | 3,793 |  12,000,265 |  941 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/keccak-f1bce711a32291920e5cc4cb28b56b371d80f98d.md) | 18,928 |  18,655,329 |  3,383 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/sha2_bench-f1bce711a32291920e5cc4cb28b56b371d80f98d.md) | 9,025 |  14,793,960 |  1,402 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/regex-f1bce711a32291920e5cc4cb28b56b371d80f98d.md) | 1,434 |  4,137,067 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/ecrecover-f1bce711a32291920e5cc4cb28b56b371d80f98d.md) | 641 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/pairing-f1bce711a32291920e5cc4cb28b56b371d80f98d.md) | 901 |  1,745,757 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/kitchen_sink-f1bce711a32291920e5cc4cb28b56b371d80f98d.md) | 2,088 |  2,579,903 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f1bce711a32291920e5cc4cb28b56b371d80f98d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24794818463)
