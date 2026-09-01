| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/fibonacci-fd3dd6368ff7d3b9c0f68debae39340fce64e07f.md) |<span style='color: red'>(+36 [+2.2%])</span> 1,703 |  12,000,265 | <span style='color: red'>(+7 [+1.9%])</span> 375 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/keccak-fd3dd6368ff7d3b9c0f68debae39340fce64e07f.md) |<span style='color: green'>(-98 [-1.0%])</span> 9,541 |  18,655,329 | <span style='color: green'>(-14 [-0.9%])</span> 1,537 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/sha2_bench-fd3dd6368ff7d3b9c0f68debae39340fce64e07f.md) |<span style='color: green'>(-19 [-0.4%])</span> 5,257 |  14,793,960 | <span style='color: green'>(-3 [-0.5%])</span> 593 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/regex-fd3dd6368ff7d3b9c0f68debae39340fce64e07f.md) |<span style='color: red'>(+17 [+2.5%])</span> 703 |  4,137,067 | <span style='color: red'>(+10 [+4.7%])</span> 225 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/ecrecover-fd3dd6368ff7d3b9c0f68debae39340fce64e07f.md) |<span style='color: green'>(-6 [-1.3%])</span> 441 |  123,583 | <span style='color: green'>(-2 [-1.0%])</span> 190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/pairing-fd3dd6368ff7d3b9c0f68debae39340fce64e07f.md) |<span style='color: red'>(+27 [+4.6%])</span> 610 |  1,745,757 | <span style='color: green'>(-3 [-1.5%])</span> 195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/kitchen_sink-fd3dd6368ff7d3b9c0f68debae39340fce64e07f.md) |<span style='color: green'>(-15 [-0.7%])</span> 2,288 |  2,579,903 | <span style='color: green'>(-5 [-1.0%])</span> 491 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fd3dd6368ff7d3b9c0f68debae39340fce64e07f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33540142421)
