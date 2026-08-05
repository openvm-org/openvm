| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/fibonacci-880e7d475480ac1fd641e49ead337933d836b510.md) | 473 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/keccak-880e7d475480ac1fd641e49ead337933d836b510.md) | 7,475 |  14,365,133 |  1,541 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/sha2_bench-880e7d475480ac1fd641e49ead337933d836b510.md) | 4,104 |  11,167,961 |  518 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/regex-880e7d475480ac1fd641e49ead337933d836b510.md) | 659 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/ecrecover-880e7d475480ac1fd641e49ead337933d836b510.md) | 229 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/pairing-880e7d475480ac1fd641e49ead337933d836b510.md) | 230 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/kitchen_sink-880e7d475480ac1fd641e49ead337933d836b510.md) | 2,035 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/880e7d475480ac1fd641e49ead337933d836b510

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31041238761)
