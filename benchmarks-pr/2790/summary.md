| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci-a93f8de3ef5203f246c14f998d97e7d2b7f34d8c.md) | 3,741 |  12,000,265 |  904 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/keccak-a93f8de3ef5203f246c14f998d97e7d2b7f34d8c.md) | 18,605 |  18,655,329 |  3,295 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/sha2_bench-a93f8de3ef5203f246c14f998d97e7d2b7f34d8c.md) | 10,184 |  14,793,960 |  1,464 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex-a93f8de3ef5203f246c14f998d97e7d2b7f34d8c.md) | 1,410 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover-a93f8de3ef5203f246c14f998d97e7d2b7f34d8c.md) | 603 |  123,583 |  243 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing-a93f8de3ef5203f246c14f998d97e7d2b7f34d8c.md) | 890 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink-a93f8de3ef5203f246c14f998d97e7d2b7f34d8c.md) | 1,889 |  2,579,903 |  411 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci_e2e-a93f8de3ef5203f246c14f998d97e7d2b7f34d8c.md) | 1,775 |  12,000,265 |  410 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex_e2e-a93f8de3ef5203f246c14f998d97e7d2b7f34d8c.md) | 817 |  4,137,067 |  172 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover_e2e-a93f8de3ef5203f246c14f998d97e7d2b7f34d8c.md) | 512 |  123,583 |  133 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing_e2e-a93f8de3ef5203f246c14f998d97e7d2b7f34d8c.md) | 637 |  1,745,757 |  130 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink_e2e-a93f8de3ef5203f246c14f998d97e7d2b7f34d8c.md) | 2,035 |  2,579,903 |  401 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a93f8de3ef5203f246c14f998d97e7d2b7f34d8c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26469184516)
