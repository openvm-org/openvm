| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/fibonacci-483e1b97a2fdac9c81622cc39a862a3a139647c6.md) |<span style='color: green'>(-15 [-0.9%])</span> 1,576 |  12,000,265 |  359 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/keccak-483e1b97a2fdac9c81622cc39a862a3a139647c6.md) |<span style='color: green'>(-57 [-0.6%])</span> 9,342 |  18,655,329 |  1,531 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/sha2_bench-483e1b97a2fdac9c81622cc39a862a3a139647c6.md) |<span style='color: green'>(-16 [-0.3%])</span> 4,983 |  14,793,960 | <span style='color: green'>(-7 [-1.2%])</span> 572 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/regex-483e1b97a2fdac9c81622cc39a862a3a139647c6.md) |<span style='color: red'>(+3 [+0.5%])</span> 664 |  4,137,067 | <span style='color: green'>(-2 [-0.9%])</span> 212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/ecrecover-483e1b97a2fdac9c81622cc39a862a3a139647c6.md) |<span style='color: red'>(+15 [+3.5%])</span> 439 |  123,583 | <span style='color: red'>(+3 [+1.6%])</span> 186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/pairing-483e1b97a2fdac9c81622cc39a862a3a139647c6.md) |<span style='color: red'>(+7 [+1.3%])</span> 560 |  1,745,757 | <span style='color: red'>(+7 [+3.7%])</span> 196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3115/kitchen_sink-483e1b97a2fdac9c81622cc39a862a3a139647c6.md) |<span style='color: green'>(-19 [-0.9%])</span> 2,214 |  2,579,903 | <span style='color: green'>(-8 [-1.7%])</span> 474 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/483e1b97a2fdac9c81622cc39a862a3a139647c6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31209248273)
