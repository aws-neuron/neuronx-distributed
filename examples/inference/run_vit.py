import requests
from PIL import Image

from runner import TaskType
from vit.vit_runner import ViTRunner

# # 1. ViT-Base model (86M): image size 224
# model_path = "/home/ubuntu/models/vit-base-patch16-224/"
# traced_model_path = "/home/ubuntu/traced_models/vit-base-patch16-224/"

# # 2. ViT-Large model (307M): image size 224
# model_path = "/home/ubuntu/models/vit-large-patch16-224/"
# traced_model_path = "/home/ubuntu/traced_models/vit-large-patch16-224/"

# 3. ViT-Huge model (632M): image size 224
model_path = "/home/ubuntu/models/vit-huge-patch14-224/"
traced_model_path = "/home/ubuntu/traced_models/vit-huge-patch14-224/"


def get_image(image_size=224):
    url = "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_images/mlp.png"
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    image = image.resize((image_size, image_size))
    return image
    

def vit_sample():
    tp_degree = 2
    batch_size = 1
    image_size = 224

    image = get_image(image_size)
    batch_image = [image] * batch_size

    runner = ViTRunner(model_path=model_path, tokenizer_path=model_path, task_type=TaskType.IMAGE_ENC)
    
    runner.trace(
        traced_model_path=traced_model_path,
        tp_degree=tp_degree,
        batch_size=batch_size,
        is_tokenizer=False,
        enable_bucketing=False
    )
    
    # Load model weights into Neuron device
    # We will use the returned model to run accuracy and perf tests
    print("\nLoading model to Neuron device ..")
    neuron_model = runner.load_neuron_model(traced_model_path)
    
    # Confirm the traced model matches the huggingface model run on cpu
    print("\nChecking accuracy for logits ..")
    runner.check_accuracy_logits(neuron_model, batch_size, image=image)

    # Perform inference
    print("\nInference for images ...")
    outputs = runner.inference_on_neuron(batch_size, neuron_model, images=batch_image)
    print(f"Inference outputs: {outputs.shape}")

    # Now lets benchmark
    print("\nBenchmarking ..")
    runner.benchmark_sampling(neuron_model, image=image, target="image_encode")


if __name__ == "__main__":
    vit_sample()