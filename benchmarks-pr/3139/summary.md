| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/fibonacci-16c26a32be3405b294993df02632e63f7fb05f65.md) |<span style='color: green'>(-1173 [-70.0%])</span> 502 | <span style='color: green'>(-8000214 [-66.7%])</span> 4,000,051 | <span style='color: green'>(-135 [-36.3%])</span> 237 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/keccak-16c26a32be3405b294993df02632e63f7fb05f65.md) |<span style='color: green'>(-1745 [-18.4%])</span> 7,733 | <span style='color: green'>(-4290196 [-23.0%])</span> 14,365,133 | <span style='color: red'>(+104 [+6.8%])</span> 1,644 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/sha2_bench-16c26a32be3405b294993df02632e63f7fb05f65.md) |<span style='color: green'>(-817 [-15.6%])</span> 4,426 | <span style='color: green'>(-3625999 [-24.5%])</span> 11,167,961 | <span style='color: green'>(-51 [-8.7%])</span> 537 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/regex-16c26a32be3405b294993df02632e63f7fb05f65.md) |<span style='color: red'>(+90 [+13.0%])</span> 782 | <span style='color: green'>(-46411 [-1.1%])</span> 4,090,656 | <span style='color: red'>(+1 [+0.5%])</span> 220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/ecrecover-16c26a32be3405b294993df02632e63f7fb05f65.md) |<span style='color: green'>(-221 [-51.6%])</span> 207 | <span style='color: green'>(-11373 [-9.2%])</span> 112,210 | <span style='color: red'>(+3 [+1.6%])</span> 190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/pairing-16c26a32be3405b294993df02632e63f7fb05f65.md) |<span style='color: green'>(-340 [-57.7%])</span> 249 | <span style='color: green'>(-1152930 [-66.0%])</span> 592,827 | <span style='color: green'>(-23 [-11.7%])</span> 174 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/kitchen_sink-16c26a32be3405b294993df02632e63f7fb05f65.md) |<span style='color: green'>(-29 [-1.3%])</span> 2,259 | <span style='color: green'>(-599932 [-23.3%])</span> 1,979,971 | <span style='color: green'>(-21 [-4.2%])</span> 476 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/fibonacci_e2e-16c26a32be3405b294993df02632e63f7fb05f65.md) | 775 |  4,000,053 |  223 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/regex_e2e-16c26a32be3405b294993df02632e63f7fb05f65.md) | 1,018 |  4,090,658 |  203 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/ecrecover_e2e-16c26a32be3405b294993df02632e63f7fb05f65.md) | 512 |  112,212 |  179 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/pairing_e2e-16c26a32be3405b294993df02632e63f7fb05f65.md) | 552 |  592,829 |  163 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/kitchen_sink_e2e-16c26a32be3405b294993df02632e63f7fb05f65.md) | 2,473 |  1,979,973 |  472 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/16c26a32be3405b294993df02632e63f7fb05f65

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33819630415)
