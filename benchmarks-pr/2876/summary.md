| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/fibonacci-125023dcd3bda2e32aee54cbd55a235459a151d0.md) | 1,659 |  4,000,051 |  529 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/keccak-125023dcd3bda2e32aee54cbd55a235459a151d0.md) | 16,105 |  14,365,133 |  3,002 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/sha2_bench-125023dcd3bda2e32aee54cbd55a235459a151d0.md) | 10,428 |  11,167,961 |  1,948 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/regex-125023dcd3bda2e32aee54cbd55a235459a151d0.md) | 1,528 |  4,090,656 |  424 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/ecrecover-125023dcd3bda2e32aee54cbd55a235459a151d0.md) | 481 |  112,210 |  305 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/pairing-125023dcd3bda2e32aee54cbd55a235459a151d0.md) | 617 |  592,827 |  290 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/kitchen_sink-125023dcd3bda2e32aee54cbd55a235459a151d0.md) | 3,972 |  1,979,971 |  871 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/125023dcd3bda2e32aee54cbd55a235459a151d0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27435638028)
