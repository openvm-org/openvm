| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci-4c6130ee3b0e252e32e45416f8aa0fb7329253f8.md) | 3,786 |  12,000,265 |  925 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/keccak-4c6130ee3b0e252e32e45416f8aa0fb7329253f8.md) | 18,697 |  18,655,329 |  3,289 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/sha2_bench-4c6130ee3b0e252e32e45416f8aa0fb7329253f8.md) | 10,456 |  14,793,960 |  1,502 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex-4c6130ee3b0e252e32e45416f8aa0fb7329253f8.md) | 1,398 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover-4c6130ee3b0e252e32e45416f8aa0fb7329253f8.md) | 603 |  123,583 |  253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing-4c6130ee3b0e252e32e45416f8aa0fb7329253f8.md) | 889 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink-4c6130ee3b0e252e32e45416f8aa0fb7329253f8.md) | 1,895 |  2,579,903 |  413 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci_e2e-4c6130ee3b0e252e32e45416f8aa0fb7329253f8.md) | 1,774 |  12,000,265 |  406 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex_e2e-4c6130ee3b0e252e32e45416f8aa0fb7329253f8.md) | 817 |  4,137,067 |  170 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover_e2e-4c6130ee3b0e252e32e45416f8aa0fb7329253f8.md) | 508 |  123,583 |  129 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing_e2e-4c6130ee3b0e252e32e45416f8aa0fb7329253f8.md) | 633 |  1,745,757 |  131 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink_e2e-4c6130ee3b0e252e32e45416f8aa0fb7329253f8.md) | 2,020 |  2,579,903 |  400 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4c6130ee3b0e252e32e45416f8aa0fb7329253f8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26975103025)
