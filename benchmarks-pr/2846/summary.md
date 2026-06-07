| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-bf4cfbf42eaa9c1c0a1298a8a93915089aa07f42.md) | 3,743 |  12,000,265 | <span style='color: green'>(-3562 [-79.4%])</span> 924 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-bf4cfbf42eaa9c1c0a1298a8a93915089aa07f42.md) | 18,171 |  18,655,329 |  3,301 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-bf4cfbf42eaa9c1c0a1298a8a93915089aa07f42.md) | 9,871 |  14,793,960 |  1,442 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-bf4cfbf42eaa9c1c0a1298a8a93915089aa07f42.md) | 1,393 |  4,137,067 | <span style='color: green'>(-11642 [-97.0%])</span> 355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-bf4cfbf42eaa9c1c0a1298a8a93915089aa07f42.md) | 601 |  123,583 | <span style='color: green'>(-5603 [-95.7%])</span> 253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-bf4cfbf42eaa9c1c0a1298a8a93915089aa07f42.md) | 884 |  1,745,757 | <span style='color: green'>(-6121 [-95.9%])</span> 259 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-bf4cfbf42eaa9c1c0a1298a8a93915089aa07f42.md) | 3,870 |  2,579,903 |  955 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-bf4cfbf42eaa9c1c0a1298a8a93915089aa07f42.md) | 1,622 |  12,000,265 |  408 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-bf4cfbf42eaa9c1c0a1298a8a93915089aa07f42.md) | 676 |  4,137,067 |  172 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-bf4cfbf42eaa9c1c0a1298a8a93915089aa07f42.md) | 359 |  123,583 |  130 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-bf4cfbf42eaa9c1c0a1298a8a93915089aa07f42.md) | 479 |  1,745,757 |  130 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-bf4cfbf42eaa9c1c0a1298a8a93915089aa07f42.md) | 1,822 |  2,579,903 |  404 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bf4cfbf42eaa9c1c0a1298a8a93915089aa07f42

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27084903036)
