| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/fibonacci-ed0dc99936f9c138f1bd32aa2972ff68e35a0aad.md) |<span style='color: green'>(-1482 [-48.7%])</span> 1,561 |  12,000,265 | <span style='color: green'>(-318 [-47.1%])</span> 357 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/keccak-ed0dc99936f9c138f1bd32aa2972ff68e35a0aad.md) |<span style='color: green'>(-6914 [-42.4%])</span> 9,375 |  18,655,329 | <span style='color: green'>(-1493 [-49.3%])</span> 1,534 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/sha2_bench-ed0dc99936f9c138f1bd32aa2972ff68e35a0aad.md) |<span style='color: green'>(-4165 [-44.9%])</span> 5,105 |  14,793,960 | <span style='color: green'>(-555 [-49.0%])</span> 577 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/regex-ed0dc99936f9c138f1bd32aa2972ff68e35a0aad.md) |<span style='color: green'>(-510 [-43.9%])</span> 651 |  4,137,067 | <span style='color: green'>(-142 [-40.3%])</span> 210 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/ecrecover-ed0dc99936f9c138f1bd32aa2972ff68e35a0aad.md) |<span style='color: green'>(-163 [-27.2%])</span> 436 |  123,583 | <span style='color: green'>(-97 [-34.4%])</span> 185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/pairing-ed0dc99936f9c138f1bd32aa2972ff68e35a0aad.md) |<span style='color: green'>(-386 [-41.2%])</span> 552 |  1,745,757 | <span style='color: green'>(-115 [-37.2%])</span> 194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/kitchen_sink-ed0dc99936f9c138f1bd32aa2972ff68e35a0aad.md) |<span style='color: green'>(-1905 [-46.3%])</span> 2,212 |  2,579,903 | <span style='color: green'>(-400 [-45.5%])</span> 479 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/fibonacci_e2e-ed0dc99936f9c138f1bd32aa2972ff68e35a0aad.md) | 1,649 |  12,000,265 |  352 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/regex_e2e-ed0dc99936f9c138f1bd32aa2972ff68e35a0aad.md) | 788 |  4,137,067 |  209 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/ecrecover_e2e-ed0dc99936f9c138f1bd32aa2972ff68e35a0aad.md) | 495 |  123,583 |  176 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/pairing_e2e-ed0dc99936f9c138f1bd32aa2972ff68e35a0aad.md) | 634 |  1,745,757 |  179 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/kitchen_sink_e2e-ed0dc99936f9c138f1bd32aa2972ff68e35a0aad.md) | 2,705 |  2,579,903 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ed0dc99936f9c138f1bd32aa2972ff68e35a0aad

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29451490250)
