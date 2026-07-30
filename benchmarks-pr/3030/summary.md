| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/fibonacci-dc43754c33be7331d5d129698a03571347151e17.md) |<span style='color: green'>(-11 [-0.7%])</span> 1,572 |  12,000,265 | <span style='color: red'>(+1 [+0.3%])</span> 361 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/keccak-dc43754c33be7331d5d129698a03571347151e17.md) |<span style='color: red'>(+74 [+0.8%])</span> 9,329 |  18,655,329 |  1,520 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/sha2_bench-dc43754c33be7331d5d129698a03571347151e17.md) |<span style='color: red'>(+14 [+0.3%])</span> 4,930 |  14,793,960 | <span style='color: red'>(+3 [+0.5%])</span> 578 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/regex-dc43754c33be7331d5d129698a03571347151e17.md) |<span style='color: red'>(+1 [+0.2%])</span> 662 |  4,137,067 | <span style='color: red'>(+3 [+1.4%])</span> 216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/ecrecover-dc43754c33be7331d5d129698a03571347151e17.md) |<span style='color: green'>(-3 [-0.7%])</span> 434 |  123,583 | <span style='color: green'>(-2 [-1.1%])</span> 184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/pairing-dc43754c33be7331d5d129698a03571347151e17.md) |<span style='color: green'>(-35 [-6.0%])</span> 553 |  1,745,757 | <span style='color: green'>(-2 [-1.1%])</span> 188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/kitchen_sink-dc43754c33be7331d5d129698a03571347151e17.md) |<span style='color: green'>(-35 [-1.6%])</span> 2,186 |  2,579,903 | <span style='color: green'>(-6 [-1.3%])</span> 472 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dc43754c33be7331d5d129698a03571347151e17

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30569037916)
