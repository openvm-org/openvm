| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci-60f2fbbe13806244c24e3b4395bef5bb036698ad.md) | 3,783 |  12,000,265 |  920 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/keccak-60f2fbbe13806244c24e3b4395bef5bb036698ad.md) | 18,719 |  18,655,329 |  3,301 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/sha2_bench-60f2fbbe13806244c24e3b4395bef5bb036698ad.md) | 10,173 |  14,793,960 |  1,463 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex-60f2fbbe13806244c24e3b4395bef5bb036698ad.md) | 1,413 |  4,137,067 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover-60f2fbbe13806244c24e3b4395bef5bb036698ad.md) | 596 |  123,583 |  246 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing-60f2fbbe13806244c24e3b4395bef5bb036698ad.md) | 891 |  1,745,757 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink-60f2fbbe13806244c24e3b4395bef5bb036698ad.md) | 1,887 |  2,579,903 |  412 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci_e2e-60f2fbbe13806244c24e3b4395bef5bb036698ad.md) | 1,778 |  12,000,265 |  410 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex_e2e-60f2fbbe13806244c24e3b4395bef5bb036698ad.md) | 832 |  4,137,067 |  171 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover_e2e-60f2fbbe13806244c24e3b4395bef5bb036698ad.md) | 512 |  123,583 |  129 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing_e2e-60f2fbbe13806244c24e3b4395bef5bb036698ad.md) | 637 |  1,745,757 |  132 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink_e2e-60f2fbbe13806244c24e3b4395bef5bb036698ad.md) | 2,028 |  2,579,903 |  403 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/60f2fbbe13806244c24e3b4395bef5bb036698ad

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26308694065)
