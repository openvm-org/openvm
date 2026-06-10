| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2863/fibonacci-37631779ea1a355651112b562dda5bb6b4683bea.md) | 3,956 |  12,000,265 |  1,143 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2863/keccak-37631779ea1a355651112b562dda5bb6b4683bea.md) | 21,614 |  18,655,329 |  4,573 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2863/sha2_bench-37631779ea1a355651112b562dda5bb6b4683bea.md) | 9,691 |  14,793,960 |  1,854 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2863/regex-37631779ea1a355651112b562dda5bb6b4683bea.md) | 1,514 |  4,137,067 |  431 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2863/ecrecover-37631779ea1a355651112b562dda5bb6b4683bea.md) | 605 |  123,583 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2863/pairing-37631779ea1a355651112b562dda5bb6b4683bea.md) | 944 |  1,745,757 |  306 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2863/kitchen_sink-37631779ea1a355651112b562dda5bb6b4683bea.md) | 4,121 |  2,579,903 |  878 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/37631779ea1a355651112b562dda5bb6b4683bea

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27280323245)
