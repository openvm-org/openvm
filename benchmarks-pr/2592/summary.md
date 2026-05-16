| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-f08f0fe28ce3fe62d2a71ce4a7abf45ca1d42ab4.md) | 3,694 |  12,000,265 |  909 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-f08f0fe28ce3fe62d2a71ce4a7abf45ca1d42ab4.md) | 18,817 |  18,655,329 |  3,305 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-f08f0fe28ce3fe62d2a71ce4a7abf45ca1d42ab4.md) | 10,445 |  14,793,960 |  1,509 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-f08f0fe28ce3fe62d2a71ce4a7abf45ca1d42ab4.md) | 1,391 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-f08f0fe28ce3fe62d2a71ce4a7abf45ca1d42ab4.md) | 598 |  123,583 |  247 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-f08f0fe28ce3fe62d2a71ce4a7abf45ca1d42ab4.md) | 883 |  1,745,757 |  259 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-f08f0fe28ce3fe62d2a71ce4a7abf45ca1d42ab4.md) | 1,895 |  2,579,903 |  413 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-f08f0fe28ce3fe62d2a71ce4a7abf45ca1d42ab4.md) | 1,778 |  12,000,265 |  411 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-f08f0fe28ce3fe62d2a71ce4a7abf45ca1d42ab4.md) | 827 |  4,137,067 |  170 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-f08f0fe28ce3fe62d2a71ce4a7abf45ca1d42ab4.md) | 505 |  123,583 |  131 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-f08f0fe28ce3fe62d2a71ce4a7abf45ca1d42ab4.md) | 638 |  1,745,757 |  132 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-f08f0fe28ce3fe62d2a71ce4a7abf45ca1d42ab4.md) | 2,019 |  2,579,903 |  399 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f08f0fe28ce3fe62d2a71ce4a7abf45ca1d42ab4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25957008281)
