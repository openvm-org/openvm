| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-93c0710f29b0e49b6e8a540960028a65fe154ea1.md) | 4,001 |  12,000,265 | <span style='color: green'>(-3336 [-74.4%])</span> 1,150 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-93c0710f29b0e49b6e8a540960028a65fe154ea1.md) | 21,815 |  18,655,329 |  4,636 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-93c0710f29b0e49b6e8a540960028a65fe154ea1.md) | 9,496 |  14,793,960 |  1,827 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-93c0710f29b0e49b6e8a540960028a65fe154ea1.md) | 1,521 |  4,137,067 | <span style='color: green'>(-11564 [-96.4%])</span> 433 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-93c0710f29b0e49b6e8a540960028a65fe154ea1.md) | 609 |  123,583 | <span style='color: green'>(-5574 [-95.2%])</span> 282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-93c0710f29b0e49b6e8a540960028a65fe154ea1.md) | 946 |  1,745,757 | <span style='color: green'>(-6072 [-95.2%])</span> 308 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-93c0710f29b0e49b6e8a540960028a65fe154ea1.md) | 4,178 |  2,579,903 |  886 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-93c0710f29b0e49b6e8a540960028a65fe154ea1.md) | 1,714 |  12,000,265 |  495 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-93c0710f29b0e49b6e8a540960028a65fe154ea1.md) | 716 |  4,137,067 |  197 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-93c0710f29b0e49b6e8a540960028a65fe154ea1.md) | 368 |  123,583 |  141 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-93c0710f29b0e49b6e8a540960028a65fe154ea1.md) | 504 |  1,745,757 |  147 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-93c0710f29b0e49b6e8a540960028a65fe154ea1.md) | 2,176 |  2,579,903 |  384 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/93c0710f29b0e49b6e8a540960028a65fe154ea1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27235204470)
