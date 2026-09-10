| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/fibonacci-26cb902eadf623c1db357da328ac271ad08dead3.md) | 492 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/keccak-26cb902eadf623c1db357da328ac271ad08dead3.md) | 7,635 |  14,365,133 |  1,629 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/sha2_bench-26cb902eadf623c1db357da328ac271ad08dead3.md) | 4,377 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/regex-26cb902eadf623c1db357da328ac271ad08dead3.md) | 773 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/ecrecover-26cb902eadf623c1db357da328ac271ad08dead3.md) | 209 |  112,210 |  190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/pairing-26cb902eadf623c1db357da328ac271ad08dead3.md) | 250 |  592,827 |  173 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/kitchen_sink-26cb902eadf623c1db357da328ac271ad08dead3.md) | 2,258 |  1,979,971 |  477 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/26cb902eadf623c1db357da328ac271ad08dead3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/34455172921)
