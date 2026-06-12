| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/fibonacci-f61ac406524fc549f86270753e32b5b24dbb19f7.md) | 3,994 |  12,000,265 |  1,156 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/keccak-f61ac406524fc549f86270753e32b5b24dbb19f7.md) | 21,513 |  18,655,329 |  4,570 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/sha2_bench-f61ac406524fc549f86270753e32b5b24dbb19f7.md) | 9,637 |  14,793,960 |  1,852 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/regex-f61ac406524fc549f86270753e32b5b24dbb19f7.md) | 1,513 |  4,137,067 |  430 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/ecrecover-f61ac406524fc549f86270753e32b5b24dbb19f7.md) | 603 |  123,583 |  280 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/pairing-f61ac406524fc549f86270753e32b5b24dbb19f7.md) | 925 |  1,745,757 |  306 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/kitchen_sink-f61ac406524fc549f86270753e32b5b24dbb19f7.md) | 4,185 |  2,579,903 |  900 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f61ac406524fc549f86270753e32b5b24dbb19f7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27448024290)
