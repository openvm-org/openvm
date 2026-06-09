| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-032522d5c31aaa70ffe0032b1cc5f2b07f6a67e6.md) | 3,739 |  12,000,265 | <span style='color: green'>(-3568 [-79.5%])</span> 918 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-032522d5c31aaa70ffe0032b1cc5f2b07f6a67e6.md) | 18,384 |  18,655,329 |  3,358 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-032522d5c31aaa70ffe0032b1cc5f2b07f6a67e6.md) | 10,047 |  14,793,960 |  1,465 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-032522d5c31aaa70ffe0032b1cc5f2b07f6a67e6.md) | 1,385 |  4,137,067 | <span style='color: green'>(-11644 [-97.1%])</span> 353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-032522d5c31aaa70ffe0032b1cc5f2b07f6a67e6.md) | 595 |  123,583 | <span style='color: green'>(-5606 [-95.7%])</span> 250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-032522d5c31aaa70ffe0032b1cc5f2b07f6a67e6.md) | 877 |  1,745,757 | <span style='color: green'>(-6115 [-95.8%])</span> 265 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-032522d5c31aaa70ffe0032b1cc5f2b07f6a67e6.md) | 3,849 |  2,579,903 |  951 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-032522d5c31aaa70ffe0032b1cc5f2b07f6a67e6.md) | 1,618 |  12,000,265 |  406 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-032522d5c31aaa70ffe0032b1cc5f2b07f6a67e6.md) | 677 |  4,137,067 |  171 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-032522d5c31aaa70ffe0032b1cc5f2b07f6a67e6.md) | 363 |  123,583 |  131 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-032522d5c31aaa70ffe0032b1cc5f2b07f6a67e6.md) | 481 |  1,745,757 |  131 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-032522d5c31aaa70ffe0032b1cc5f2b07f6a67e6.md) | 1,824 |  2,579,903 |  399 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/032522d5c31aaa70ffe0032b1cc5f2b07f6a67e6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27198420094)
