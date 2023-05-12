# NeuronxDistributed

Source code for `neuronx-distributed`. This package is used for releasing neuron-friendly implementations 
of certain 3rd party tools.

## Development resources

VersionSet: KaenaPyTorchXLATest/development: https://code.amazon.com/version-sets/KaenaPyTorchXLATest/development

Pipeline: KaenaPyTorchXLATest: https://pipelines.amazon.com/pipelines/KaenaPyTorchXLATest

Auto Test Pipeline: https://pipelines.amazon.com/pipelines/KaenaPyTorchXLATest-development-autotest

## Building NeuronxDistributed

```
brazil ws create --name NeuronxDistributed --versionSet KaenaPyTorchXLATest/development
cd NeuronxDistributed
brazil ws use NeuronxDistributed

# Build NeuronxDistributed to get neuronx-distributed
cd src/NeuronxDistributed
git checkout mainline
bb clean && bb
```
