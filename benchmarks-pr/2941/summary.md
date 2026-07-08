| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/fibonacci-3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3.md) | 3,050 |  12,000,265 | <span style='color: green'>(-3515 [-83.7%])</span> 684 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/keccak-3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3.md) | 16,274 |  18,655,329 |  3,015 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/sha2_bench-3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3.md) | 9,248 |  14,793,960 |  1,135 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/regex-3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3.md) | 1,169 |  4,137,067 | <span style='color: green'>(-11771 [-97.1%])</span> 350 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/ecrecover-3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3.md) | 607 |  123,583 | <span style='color: green'>(-5982 [-95.4%])</span> 289 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/pairing-3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3.md) | 952 |  1,745,757 | <span style='color: green'>(-6342 [-95.3%])</span> 311 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/kitchen_sink-3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3.md) | 4,126 |  2,579,903 |  891 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/fibonacci_e2e-3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3.md) | 1,520 |  12,000,265 |  288 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/regex_e2e-3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3.md) | 761 |  4,137,067 |  167 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/ecrecover_e2e-3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3.md) | 507 |  123,583 |  143 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/pairing_e2e-3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3.md) | 652 |  1,745,757 |  147 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/kitchen_sink_e2e-3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3.md) | 2,462 |  2,579,903 |  390 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3ee99e5002e123a6ca50c815c3e40ab0bd96b0e3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28954508551)
