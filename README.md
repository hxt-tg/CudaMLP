# CudaMLP
A MLP implementation on multicore processors with Cuda.

## Test dataset

The classic IRIS dataset is used to test and verify the MLP algorithm.

Dataset is downloaded from: https://archive.ics.uci.edu/ml/datasets/iris

All files are unzipped to `_data` directory.

## Technical features

- Computational intensive
- Efficient training and fitting
- Easily portable and expandable
- Lightweight and friendly program interface

## TODO

- [x] CUDA or CPU parallel?
- [x] API design
- [x] Algebra calculation module
- [x] Data feeding module
- [x] MLP training module
- [x] Unit test

## TODO (Depend on time)
- [ ] Reinforcement learning application (e.g. DQN)
- [ ] Optimize IO with coroutine in C++20.

## License
This code is released under the Apache-2.0 License.
```text
Copyright (C) 2020
Xintao Hu <hxt.taoge@gmail.com>
```
