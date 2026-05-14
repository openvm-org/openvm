| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-6bde56f8ca039dffee5d3c1b4df149c476b5139e.md) | 1,590 |  4,000,051 |  457 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-6bde56f8ca039dffee5d3c1b4df149c476b5139e.md) | 13,994 |  14,365,133 |  2,431 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-6bde56f8ca039dffee5d3c1b4df149c476b5139e.md) | 9,244 |  11,167,961 |  1,407 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-6bde56f8ca039dffee5d3c1b4df149c476b5139e.md) | 1,503 |  4,090,656 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-6bde56f8ca039dffee5d3c1b4df149c476b5139e.md) | 511 |  112,210 |  290 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-6bde56f8ca039dffee5d3c1b4df149c476b5139e.md) | 618 |  592,827 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-6bde56f8ca039dffee5d3c1b4df149c476b5139e.md) | 1,955 |  1,979,971 |  429 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6bde56f8ca039dffee5d3c1b4df149c476b5139e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25885548904)
