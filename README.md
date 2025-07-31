# Neuron Distributed

Neuron Distributed is a package for supporting different distributed training/inference mechanism for Neuron devices. It would provide xla friendly implementations of some of the more popular distributed training/inference techniques. As the size of the model scales, fitting these models on a single device becomes impossible and hence we have to make use of model sharding techniques to partition the model across multiple devices. As part of this library, we enable support for Tensor Parallel sharding technique with other distributed library supported to be added in future.

## Building/Installing NeuronxDistributed

To install the library, please follow the instructions mentioned here: https://awsdocs-neuron.readthedocs-hosted.com/en/latest//libraries/neuronx-distributed/index.html

To build from source, run the following command:

```
bash ./build.sh
```

It should place the wheel at `build/`

## API Reference Guide

For a detailed API reference guide, please refer to: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/api_guide.html#api-guide


### Formatting code

To format the code, use the following command:
```
pre-commit run --all-files
```
