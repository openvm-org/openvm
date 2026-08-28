| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/fibonacci-63791d01347e46c89003ce969fc825aabbcf516c.md) |<span style='color: green'>(-1190 [-70.2%])</span> 505 | <span style='color: green'>(-8000214 [-66.7%])</span> 4,000,051 | <span style='color: green'>(-133 [-35.8%])</span> 239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/keccak-63791d01347e46c89003ce969fc825aabbcf516c.md) |<span style='color: green'>(-2038 [-21.4%])</span> 7,500 | <span style='color: green'>(-4290196 [-23.0%])</span> 14,365,133 | <span style='color: red'>(+32 [+2.1%])</span> 1,577 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/sha2_bench-63791d01347e46c89003ce969fc825aabbcf516c.md) |<span style='color: green'>(-923 [-17.6%])</span> 4,320 | <span style='color: green'>(-3625999 [-24.5%])</span> 11,167,961 | <span style='color: green'>(-62 [-10.6%])</span> 524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/regex-63791d01347e46c89003ce969fc825aabbcf516c.md) |<span style='color: red'>(+74 [+10.4%])</span> 783 | <span style='color: green'>(-46411 [-1.1%])</span> 4,090,656 | <span style='color: green'>(-1 [-0.5%])</span> 218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/ecrecover-63791d01347e46c89003ce969fc825aabbcf516c.md) |<span style='color: green'>(-229 [-51.8%])</span> 213 | <span style='color: green'>(-11373 [-9.2%])</span> 112,210 | <span style='color: red'>(+3 [+1.6%])</span> 193 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/pairing-63791d01347e46c89003ce969fc825aabbcf516c.md) |<span style='color: green'>(-337 [-57.1%])</span> 253 | <span style='color: green'>(-1152930 [-66.0%])</span> 592,827 | <span style='color: green'>(-21 [-10.7%])</span> 175 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/kitchen_sink-63791d01347e46c89003ce969fc825aabbcf516c.md) |<span style='color: green'>(-42 [-1.8%])</span> 2,248 | <span style='color: green'>(-599932 [-23.3%])</span> 1,979,971 | <span style='color: green'>(-24 [-4.8%])</span> 473 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/fibonacci_e2e-63791d01347e46c89003ce969fc825aabbcf516c.md) | 779 |  4,000,053 |  227 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/regex_e2e-63791d01347e46c89003ce969fc825aabbcf516c.md) | 1,094 |  4,090,658 |  208 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/ecrecover_e2e-63791d01347e46c89003ce969fc825aabbcf516c.md) | 514 |  112,212 |  176 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/pairing_e2e-63791d01347e46c89003ce969fc825aabbcf516c.md) | 551 |  592,829 |  163 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/kitchen_sink_e2e-63791d01347e46c89003ce969fc825aabbcf516c.md) | 2,464 |  1,979,973 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/63791d01347e46c89003ce969fc825aabbcf516c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33203799282)
