| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2996/fibonacci-b6f63a5ce389afc92afb17485ebe283fa9b4bf32.md) | 3,029 |  12,000,265 |  682 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2996/keccak-b6f63a5ce389afc92afb17485ebe283fa9b4bf32.md) | 16,689 |  18,655,329 |  3,085 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2996/sha2_bench-b6f63a5ce389afc92afb17485ebe283fa9b4bf32.md) | 9,358 |  14,793,960 |  1,123 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2996/regex-b6f63a5ce389afc92afb17485ebe283fa9b4bf32.md) | 1,204 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2996/ecrecover-b6f63a5ce389afc92afb17485ebe283fa9b4bf32.md) | 513 |  123,583 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2996/pairing-b6f63a5ce389afc92afb17485ebe283fa9b4bf32.md) | 847 |  1,745,757 |  311 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2996/kitchen_sink-b6f63a5ce389afc92afb17485ebe283fa9b4bf32.md) | 4,505 |  2,579,903 |  880 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b6f63a5ce389afc92afb17485ebe283fa9b4bf32

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29062646120)
