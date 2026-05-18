| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/fibonacci-e5b5099baa52b6872c099774ae178e9fec2c27d3.md) | 1,846 |  4,000,051 |  438 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/keccak-e5b5099baa52b6872c099774ae178e9fec2c27d3.md) | 14,059 |  14,365,133 |  2,228 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/sha2_bench-e5b5099baa52b6872c099774ae178e9fec2c27d3.md) | 8,068 |  11,167,961 |  900 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/regex-e5b5099baa52b6872c099774ae178e9fec2c27d3.md) | 1,535 |  4,090,656 |  363 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/ecrecover-e5b5099baa52b6872c099774ae178e9fec2c27d3.md) | 611 |  112,210 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/pairing-e5b5099baa52b6872c099774ae178e9fec2c27d3.md) | 725 |  592,827 |  260 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/kitchen_sink-e5b5099baa52b6872c099774ae178e9fec2c27d3.md) | 1,881 |  1,979,971 |  407 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e5b5099baa52b6872c099774ae178e9fec2c27d3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26056808591)
