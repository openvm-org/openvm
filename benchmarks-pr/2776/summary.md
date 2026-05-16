| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/fibonacci-7bc0e850974ea7a637ddfdfc7b73cabc8487af1d.md) | 3,748 |  12,000,265 |  911 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/keccak-7bc0e850974ea7a637ddfdfc7b73cabc8487af1d.md) | 18,672 |  18,655,329 |  3,292 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/sha2_bench-7bc0e850974ea7a637ddfdfc7b73cabc8487af1d.md) | 10,218 |  14,793,960 |  1,460 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/regex-7bc0e850974ea7a637ddfdfc7b73cabc8487af1d.md) | 1,386 |  4,137,067 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/ecrecover-7bc0e850974ea7a637ddfdfc7b73cabc8487af1d.md) | 591 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/pairing-7bc0e850974ea7a637ddfdfc7b73cabc8487af1d.md) | 896 |  1,745,757 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/kitchen_sink-7bc0e850974ea7a637ddfdfc7b73cabc8487af1d.md) | 1,886 |  2,579,903 |  411 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7bc0e850974ea7a637ddfdfc7b73cabc8487af1d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25954384633)
