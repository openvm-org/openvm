| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3127/fibonacci-4b6eca7157d2fd807ed86366b3238827417808b2.md) |<span style='color: red'>(+89 [+5.7%])</span> 1,659 |  12,000,265 | <span style='color: red'>(+12 [+3.3%])</span> 371 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3127/keccak-4b6eca7157d2fd807ed86366b3238827417808b2.md) |<span style='color: red'>(+270 [+2.9%])</span> 9,592 |  18,655,329 |  1,535 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3127/sha2_bench-4b6eca7157d2fd807ed86366b3238827417808b2.md) |<span style='color: red'>(+474 [+9.7%])</span> 5,365 |  14,793,960 | <span style='color: red'>(+19 [+3.3%])</span> 598 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3127/regex-4b6eca7157d2fd807ed86366b3238827417808b2.md) |<span style='color: red'>(+39 [+5.9%])</span> 695 |  4,137,067 | <span style='color: red'>(+7 [+3.3%])</span> 220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3127/ecrecover-4b6eca7157d2fd807ed86366b3238827417808b2.md) |<span style='color: green'>(-1 [-0.2%])</span> 434 |  123,583 | <span style='color: red'>(+3 [+1.6%])</span> 188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3127/pairing-4b6eca7157d2fd807ed86366b3238827417808b2.md) |<span style='color: red'>(+27 [+4.8%])</span> 588 |  1,745,757 |  194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3127/kitchen_sink-4b6eca7157d2fd807ed86366b3238827417808b2.md) |<span style='color: red'>(+97 [+4.4%])</span> 2,290 |  2,579,903 | <span style='color: red'>(+31 [+6.6%])</span> 503 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4b6eca7157d2fd807ed86366b3238827417808b2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32517711294)
