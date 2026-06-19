| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-514ec14abe88de6bed7352ed94a094f414d913b2.md) | 3,012 |  12,000,265 | <span style='color: green'>(-3817 [-85.1%])</span> 669 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-514ec14abe88de6bed7352ed94a094f414d913b2.md) | 16,790 |  18,655,329 |  3,108 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-514ec14abe88de6bed7352ed94a094f414d913b2.md) | 9,231 |  14,793,960 |  1,118 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-514ec14abe88de6bed7352ed94a094f414d913b2.md) | 1,167 |  4,137,067 | <span style='color: green'>(-11646 [-97.1%])</span> 351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-514ec14abe88de6bed7352ed94a094f414d913b2.md) | 598 |  123,583 | <span style='color: green'>(-5574 [-95.2%])</span> 282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-514ec14abe88de6bed7352ed94a094f414d913b2.md) | 949 |  1,745,757 | <span style='color: green'>(-6067 [-95.1%])</span> 313 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-514ec14abe88de6bed7352ed94a094f414d913b2.md) | 4,123 |  2,579,903 |  883 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-514ec14abe88de6bed7352ed94a094f414d913b2.md) | 1,394 |  12,000,265 |  284 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-514ec14abe88de6bed7352ed94a094f414d913b2.md) | 643 |  4,137,067 |  166 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-514ec14abe88de6bed7352ed94a094f414d913b2.md) | 394 |  123,583 |  143 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-514ec14abe88de6bed7352ed94a094f414d913b2.md) | 527 |  1,745,757 |  148 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-514ec14abe88de6bed7352ed94a094f414d913b2.md) | 1,989 |  2,579,903 |  385 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/514ec14abe88de6bed7352ed94a094f414d913b2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27819976726)
