| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/fibonacci-d8abb939b9d71406224cfee2b72ee909c1dc0ec8.md) |<span style='color: green'>(-25 [-1.5%])</span> 1,654 |  12,000,265 | <span style='color: green'>(-6 [-1.6%])</span> 365 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/keccak-d8abb939b9d71406224cfee2b72ee909c1dc0ec8.md) |<span style='color: green'>(-149 [-1.5%])</span> 9,501 |  18,655,329 | <span style='color: green'>(-33 [-2.1%])</span> 1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/sha2_bench-d8abb939b9d71406224cfee2b72ee909c1dc0ec8.md) |<span style='color: red'>(+24 [+0.5%])</span> 5,332 |  14,793,960 | <span style='color: green'>(-3 [-0.5%])</span> 589 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/regex-d8abb939b9d71406224cfee2b72ee909c1dc0ec8.md) |<span style='color: green'>(-19 [-2.7%])</span> 683 |  4,137,067 | <span style='color: red'>(+1 [+0.5%])</span> 217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/ecrecover-d8abb939b9d71406224cfee2b72ee909c1dc0ec8.md) |<span style='color: green'>(-14 [-3.2%])</span> 429 |  123,583 | <span style='color: green'>(-5 [-2.6%])</span> 188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/pairing-d8abb939b9d71406224cfee2b72ee909c1dc0ec8.md) |<span style='color: red'>(+26 [+4.5%])</span> 608 |  1,745,757 |  194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3176/kitchen_sink-d8abb939b9d71406224cfee2b72ee909c1dc0ec8.md) |<span style='color: red'>(+4 [+0.2%])</span> 2,320 |  2,579,903 | <span style='color: red'>(+3 [+0.6%])</span> 501 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d8abb939b9d71406224cfee2b72ee909c1dc0ec8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/40376839029)
