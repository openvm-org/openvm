| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/fibonacci-13601c2b85dcd92ff176bc264e42094dece145fc.md) | 1,887 |  4,000,051 |  535 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/keccak-13601c2b85dcd92ff176bc264e42094dece145fc.md) | 13,644 |  14,365,133 |  2,250 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/sha2_bench-13601c2b85dcd92ff176bc264e42094dece145fc.md) | 9,583 |  11,167,961 |  1,423 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/regex-13601c2b85dcd92ff176bc264e42094dece145fc.md) | 1,598 |  4,090,656 |  379 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/ecrecover-13601c2b85dcd92ff176bc264e42094dece145fc.md) | 638 |  112,210 |  289 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/pairing-13601c2b85dcd92ff176bc264e42094dece145fc.md) | 766 |  592,827 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/kitchen_sink-13601c2b85dcd92ff176bc264e42094dece145fc.md) | 2,034 |  1,979,971 |  431 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/13601c2b85dcd92ff176bc264e42094dece145fc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25764223013)
