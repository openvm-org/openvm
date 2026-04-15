| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-2a70f589f134d26d3b717e3133994bb35a4830ed.md) | 3,825 |  12,000,265 |  950 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-2a70f589f134d26d3b717e3133994bb35a4830ed.md) | 18,998 |  18,655,329 |  3,401 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-2a70f589f134d26d3b717e3133994bb35a4830ed.md) | 9,096 |  14,793,960 |  1,408 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-2a70f589f134d26d3b717e3133994bb35a4830ed.md) | 1,438 |  4,137,067 |  384 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-2a70f589f134d26d3b717e3133994bb35a4830ed.md) | 644 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-2a70f589f134d26d3b717e3133994bb35a4830ed.md) | 905 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-2a70f589f134d26d3b717e3133994bb35a4830ed.md) | 2,086 |  2,579,903 |  435 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-2a70f589f134d26d3b717e3133994bb35a4830ed.md) | 1,864 |  12,000,265 |  452 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-2a70f589f134d26d3b717e3133994bb35a4830ed.md) | 851 |  4,137,067 |  191 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-2a70f589f134d26d3b717e3133994bb35a4830ed.md) | 554 |  123,583 |  152 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-2a70f589f134d26d3b717e3133994bb35a4830ed.md) | 660 |  1,745,757 |  155 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-2a70f589f134d26d3b717e3133994bb35a4830ed.md) | 2,223 |  2,579,903 |  426 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2a70f589f134d26d3b717e3133994bb35a4830ed

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24467747641)
