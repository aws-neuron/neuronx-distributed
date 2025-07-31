# Example: Llama3.2-1B Inference

This is sample code to show how you can use Neuronx Distributed (NxD) to run a Llama3.2 1B model. 

The model is defined in `model.py`.

`run.py` has functions that you can use to:
1. Run the model on CPU
2. Compile the model (with three different approaches)
3. Run the model on Neuron (with flexible weight loading strategies)

We recommend reading the code, especially the comments to understand how the modeling
code works and runs on Neuron.

## How do I run this sample?

1. Install `neuronx-distributed`.
2. Install requirements: `pip install -r requirements.txt`
3. Download `Llama3.2-1B-Instruct` from https://www.llama.com/llama-downloads/.

Note: This sample is tested with various batch sizes and sequence lengths on Trn1 with TP degrees 1, 8, 16, and 32.

## Run on CPU

Command
```
python run.py generate_cpu \
  --batch-size 2 \
  --seq-len 128 \
  --model-path ~/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth \
  --tokenizer-path ~/.llama/checkpoints/Llama3.2-1B-Instruct/tokenizer.model \
  --prompts "['How tall is the Space Needle?','What is the capital of France?']"
```

Output
```
<|begin_of_text|>How tall is the Space Needle? 605 feet
The Space Needle is 605 feet tall. It is a 605-foot-tall tower located in Seattle, Washington, and it was built for the 1962 World's Fair. The tower was designed by architect John Graham Jr. and engineer Victor Steinbrueck, and it was completed in 1962. The Space Needle is a popular tourist attraction and a iconic symbol of Seattle.<|eot_id|>
<|begin_of_text|>What is the capital of France? Paris.
The capital of France is Paris. Paris is the most populous city in France and is known for its rich history, art, fashion, and cuisine. It is also home to many famous landmarks such as the Eiffel Tower, Notre Dame Cathedral, and the Louvre Museum. Paris is a popular tourist destination and is often referred to as the "City of Light."<|eot_id|>
```
## Compile Model 

There are three ways to compile the model:

### 1. Using `NxDParallelState` (Recommended)

Uses a context manager that handles distributed environment setup and cleanup automatically.

Command
```
python run.py compile \
  --tp-degree 32 \
  --batch-size 2 \
  --seq-len 128 \
  --model-path ~/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth \
  --output-path ~/neuron_models/Llama3.2-1B-Instruct \
  --shard-on-load True
```

### 2. Using Mock Distributed

Uses a mock distributed environment for compilation.

Command
```
python run.py compile_with_mock \
  --tp-degree 32 \
  --batch-size 2 \
  --seq-len 128 \
  --model-path ~/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth \
  --output-path ~/neuron_models/Llama3.2-1B-Instruct \
  --shard-on-load True
```

### 3. Using Actual Torch Distributed

Uses `torchrun` to launch distributed processes.

Command
```
python run.py compile_no_mock \
  --tp-degree 32 \
  --batch-size 2 \
  --seq-len 128 \
  --model-path ~/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth \
  --output-path ~/neuron_models/Llama3.2-1B-Instruct \
  --shard-on-load True
```

Output
```
Saved compiled model to ~/neuron_models/Llama3.2-1B-Instruct/
```

## Weight Loading Strategies

The `shard-on-load` parameter used in compile controls how model weights are handled:

- `shard-on-load=True` (default): Weights are sharded during model loading
- `shard-on-load=False`: Weights are pre-sharded during compilation and saved to disk

### Output Directory Structure Comparison

With `shard-on-load=True`:

```
~/neuron_models/Llama3.2-1B-Instruct/ 
│
├── config.json           # Contains model config
└── nxd_model.pt          # Compiled model
```

With `shard-on-load=False`:

```
~/neuron_models/Llama3.2-1B-Instruct/
│
├── config.json          # Contains model config
├── nxd_model.pt         # Compiled model
└── weights/             # Sharded checkpoints
    ├── tp0_sharded_checkpoint.safetensors
    ├── tp1_sharded_checkpoint.safetensors
    ├── ...
    └── tp31_sharded_checkpoint.safetensors
```

## Run model on Neuron 

Command
```
python run.py generate_nxd \
  --compiled-model-path ~/neuron_models/Llama3.2-1B-Instruct \
  --model-path ~/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth \
  --prompts "['How tall is the Space Needle?','What is the capital of France?']"
```

Output
```
<|begin_of_text|>How tall is the Space Needle? 605 feet
The Space Needle is 605 feet tall. It is a 605-foot-tall tower located in Seattle, Washington, and it was built for the 1962 World's Fair. The tower was designed by architect John Graham Jr. and engineer Victor Steinbrueck, and it was completed in 1962. The Space Needle is a popular tourist attraction and a iconic symbol of Seattle.<|eot_id|>
<|begin_of_text|>What is the capital of France? Paris.
The capital of France is Paris. Paris is the most populous city in France and is known for its rich history, art, fashion, and cuisine. It is also home to many famous landmarks such as the Eiffel Tower, Notre Dame Cathedral, and the Louvre Museum. Paris is a popular tourist destination and is often referred to as the "City of Light."<|eot_id|>
```

## Testing

To test the attention module specifically:

```
python run.py test_attention
```