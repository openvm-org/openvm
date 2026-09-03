| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3148/fibonacci-070bfcd45c3a8a1b4b8b3e302fe7257225a297d4.md) |<span style='color: green'>(-16 [-1.0%])</span> 1,665 |  12,000,265 | <span style='color: green'>(-6 [-1.6%])</span> 367 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3148/keccak-070bfcd45c3a8a1b4b8b3e302fe7257225a297d4.md) |<span style='color: red'>(+35 [+0.4%])</span> 9,721 |  18,655,329 | <span style='color: red'>(+3 [+0.2%])</span> 1,563 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3148/sha2_bench-070bfcd45c3a8a1b4b8b3e302fe7257225a297d4.md) |<span style='color: green'>(-34 [-0.6%])</span> 5,338 |  14,793,960 | <span style='color: green'>(-5 [-0.8%])</span> 592 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3148/regex-070bfcd45c3a8a1b4b8b3e302fe7257225a297d4.md) |<span style='color: green'>(-2 [-0.3%])</span> 701 |  4,137,067 | <span style='color: red'>(+1 [+0.5%])</span> 219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3148/ecrecover-070bfcd45c3a8a1b4b8b3e302fe7257225a297d4.md) |<span style='color: red'>(+5 [+1.1%])</span> 440 |  123,583 | <span style='color: red'>(+2 [+1.1%])</span> 192 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3148/pairing-070bfcd45c3a8a1b4b8b3e302fe7257225a297d4.md) |<span style='color: red'>(+8 [+1.4%])</span> 596 |  1,745,757 | <span style='color: red'>(+1 [+0.5%])</span> 197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3148/kitchen_sink-070bfcd45c3a8a1b4b8b3e302fe7257225a297d4.md) |<span style='color: red'>(+30 [+1.3%])</span> 2,322 |  2,579,903 | <span style='color: red'>(+2 [+0.4%])</span> 496 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/070bfcd45c3a8a1b4b8b3e302fe7257225a297d4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33778440141)
