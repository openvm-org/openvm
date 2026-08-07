| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/fibonacci-483e1b97a2fdac9c81622cc39a862a3a139647c6.md) |<span style='color: green'>(-14 [-0.9%])</span> 1,577 |  12,000,265 | <span style='color: red'>(+5 [+1.4%])</span> 364 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/keccak-483e1b97a2fdac9c81622cc39a862a3a139647c6.md) |<span style='color: green'>(-62 [-0.7%])</span> 9,337 |  18,655,329 | <span style='color: red'>(+3 [+0.2%])</span> 1,534 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/sha2_bench-483e1b97a2fdac9c81622cc39a862a3a139647c6.md) |<span style='color: green'>(-33 [-0.7%])</span> 4,966 |  14,793,960 | <span style='color: red'>(+3 [+0.5%])</span> 582 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/regex-483e1b97a2fdac9c81622cc39a862a3a139647c6.md) | 661 |  4,137,067 | <span style='color: red'>(+1 [+0.5%])</span> 215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/ecrecover-483e1b97a2fdac9c81622cc39a862a3a139647c6.md) |<span style='color: red'>(+10 [+2.4%])</span> 434 |  123,583 | <span style='color: red'>(+1 [+0.5%])</span> 184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/pairing-483e1b97a2fdac9c81622cc39a862a3a139647c6.md) |<span style='color: red'>(+25 [+4.5%])</span> 578 |  1,745,757 | <span style='color: red'>(+1 [+0.5%])</span> 190 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/kitchen_sink-483e1b97a2fdac9c81622cc39a862a3a139647c6.md) |<span style='color: green'>(-11 [-0.5%])</span> 2,222 |  2,579,903 | <span style='color: green'>(-4 [-0.8%])</span> 478 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/483e1b97a2fdac9c81622cc39a862a3a139647c6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31215511695)
