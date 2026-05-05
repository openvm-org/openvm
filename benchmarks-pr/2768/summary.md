| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/fibonacci-e4d60b4c3333a5fd4de5d46f4dfa7121e00850b4.md) | 4,386 |  12,000,265 |  1,349 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/keccak-e4d60b4c3333a5fd4de5d46f4dfa7121e00850b4.md) | 21,778 |  18,655,329 |  4,019 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/sha2_bench-e4d60b4c3333a5fd4de5d46f4dfa7121e00850b4.md) | 10,849 |  14,793,960 |  1,713 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/regex-e4d60b4c3333a5fd4de5d46f4dfa7121e00850b4.md) | 1,590 |  4,137,067 |  475 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/ecrecover-e4d60b4c3333a5fd4de5d46f4dfa7121e00850b4.md) | 674 |  123,583 |  355 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/pairing-e4d60b4c3333a5fd4de5d46f4dfa7121e00850b4.md) | 1,007 |  1,745,757 |  372 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/kitchen_sink-e4d60b4c3333a5fd4de5d46f4dfa7121e00850b4.md) | 2,321 |  2,579,903 |  656 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/fibonacci_e2e-e4d60b4c3333a5fd4de5d46f4dfa7121e00850b4.md) | 2,019 |  12,000,265 |  598 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/regex_e2e-e4d60b4c3333a5fd4de5d46f4dfa7121e00850b4.md) | 911 |  4,137,067 |  239 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/ecrecover_e2e-e4d60b4c3333a5fd4de5d46f4dfa7121e00850b4.md) | 559 |  123,583 |  187 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/pairing_e2e-e4d60b4c3333a5fd4de5d46f4dfa7121e00850b4.md) | 684 |  1,745,757 |  186 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2768/kitchen_sink_e2e-e4d60b4c3333a5fd4de5d46f4dfa7121e00850b4.md) | 2,447 |  2,579,903 |  649 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e4d60b4c3333a5fd4de5d46f4dfa7121e00850b4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25387818431)
