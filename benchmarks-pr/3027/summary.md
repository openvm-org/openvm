| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/fibonacci-54654e2b86c8c00eb86cb7243d6e0cabf4e8046b.md) |<span style='color: green'>(-1465 [-48.1%])</span> 1,578 |  12,000,265 | <span style='color: green'>(-315 [-46.7%])</span> 360 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/keccak-54654e2b86c8c00eb86cb7243d6e0cabf4e8046b.md) |<span style='color: green'>(-7059 [-43.3%])</span> 9,230 |  18,655,329 | <span style='color: green'>(-1507 [-49.8%])</span> 1,520 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/sha2_bench-54654e2b86c8c00eb86cb7243d6e0cabf4e8046b.md) |<span style='color: green'>(-4393 [-47.4%])</span> 4,877 |  14,793,960 | <span style='color: green'>(-553 [-48.9%])</span> 579 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/regex-54654e2b86c8c00eb86cb7243d6e0cabf4e8046b.md) |<span style='color: green'>(-494 [-42.5%])</span> 667 |  4,137,067 | <span style='color: green'>(-135 [-38.4%])</span> 217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/ecrecover-54654e2b86c8c00eb86cb7243d6e0cabf4e8046b.md) |<span style='color: green'>(-176 [-29.4%])</span> 423 |  123,583 | <span style='color: green'>(-96 [-34.0%])</span> 186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/pairing-54654e2b86c8c00eb86cb7243d6e0cabf4e8046b.md) |<span style='color: green'>(-383 [-40.8%])</span> 555 |  1,745,757 | <span style='color: green'>(-118 [-38.2%])</span> 191 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/kitchen_sink-54654e2b86c8c00eb86cb7243d6e0cabf4e8046b.md) |<span style='color: green'>(-1928 [-46.8%])</span> 2,189 |  2,579,903 | <span style='color: green'>(-404 [-46.0%])</span> 475 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/fibonacci_e2e-54654e2b86c8c00eb86cb7243d6e0cabf4e8046b.md) | 1,617 |  12,000,265 |  348 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/regex_e2e-54654e2b86c8c00eb86cb7243d6e0cabf4e8046b.md) | 770 |  4,137,067 |  201 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/ecrecover_e2e-54654e2b86c8c00eb86cb7243d6e0cabf4e8046b.md) | 503 |  123,583 |  176 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/pairing_e2e-54654e2b86c8c00eb86cb7243d6e0cabf4e8046b.md) | 661 |  1,745,757 |  185 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/kitchen_sink_e2e-54654e2b86c8c00eb86cb7243d6e0cabf4e8046b.md) | 2,739 |  2,579,903 |  471 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/54654e2b86c8c00eb86cb7243d6e0cabf4e8046b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29455384508)
