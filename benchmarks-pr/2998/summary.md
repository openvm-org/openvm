| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2998/fibonacci-d16a717279e467338bae3eec57973c8ce6698394.md) |<span style='color: red'>(+19 [+0.6%])</span> 3,021 |  12,000,265 | <span style='color: red'>(+10 [+1.5%])</span> 669 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2998/keccak-d16a717279e467338bae3eec57973c8ce6698394.md) |<span style='color: red'>(+79 [+0.5%])</span> 16,464 |  18,655,329 | <span style='color: red'>(+16 [+0.5%])</span> 3,051 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2998/sha2_bench-d16a717279e467338bae3eec57973c8ce6698394.md) |<span style='color: red'>(+54 [+0.6%])</span> 9,298 |  14,793,960 | <span style='color: red'>(+3 [+0.3%])</span> 1,130 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2998/regex-d16a717279e467338bae3eec57973c8ce6698394.md) |<span style='color: red'>(+16 [+1.4%])</span> 1,177 |  4,137,067 | <span style='color: red'>(+6 [+1.7%])</span> 358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2998/ecrecover-d16a717279e467338bae3eec57973c8ce6698394.md) |<span style='color: red'>(+9 [+1.5%])</span> 603 |  123,583 | <span style='color: red'>(+4 [+1.4%])</span> 288 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2998/pairing-d16a717279e467338bae3eec57973c8ce6698394.md) |<span style='color: green'>(-8 [-0.9%])</span> 933 |  1,745,757 | <span style='color: green'>(-4 [-1.3%])</span> 304 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2998/kitchen_sink-d16a717279e467338bae3eec57973c8ce6698394.md) |<span style='color: green'>(-17 [-0.4%])</span> 4,178 |  2,579,903 | <span style='color: red'>(+10 [+1.1%])</span> 914 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d16a717279e467338bae3eec57973c8ce6698394

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29068852424)
