| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-353de1c38871057a86a356993253d5fb540d0528.md) | 1,583 |  12,000,265 |  362 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-353de1c38871057a86a356993253d5fb540d0528.md) | 9,378 |  18,655,329 |  1,551 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-353de1c38871057a86a356993253d5fb540d0528.md) | 4,959 |  14,793,960 |  578 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-353de1c38871057a86a356993253d5fb540d0528.md) | 664 |  4,137,067 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-353de1c38871057a86a356993253d5fb540d0528.md) | 434 |  123,583 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-353de1c38871057a86a356993253d5fb540d0528.md) | 556 |  1,745,757 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-353de1c38871057a86a356993253d5fb540d0528.md) | 2,215 |  2,579,903 |  473 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/353de1c38871057a86a356993253d5fb540d0528

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30570777749)
