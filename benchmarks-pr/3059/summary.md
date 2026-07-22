| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3059/fibonacci-7889acc6f0990ba28f34121f178e202e0b91b4a4.md) | 468 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3059/keccak-7889acc6f0990ba28f34121f178e202e0b91b4a4.md) | 7,335 |  14,365,133 |  1,532 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3059/sha2_bench-7889acc6f0990ba28f34121f178e202e0b91b4a4.md) | 4,691 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3059/regex-7889acc6f0990ba28f34121f178e202e0b91b4a4.md) | 671 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3059/ecrecover-7889acc6f0990ba28f34121f178e202e0b91b4a4.md) | 228 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3059/pairing-7889acc6f0990ba28f34121f178e202e0b91b4a4.md) | 333 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3059/kitchen_sink-7889acc6f0990ba28f34121f178e202e0b91b4a4.md) | 2,662 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7889acc6f0990ba28f34121f178e202e0b91b4a4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29940876687)
