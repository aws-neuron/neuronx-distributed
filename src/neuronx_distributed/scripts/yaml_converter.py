import json
import yaml


def load_yaml_file(file_path):
    try:
        with open(file_path, "r") as file:
            yaml_data = yaml.safe_load(file)
        return yaml_data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None


def convert_yaml_to_json(yaml_path, filename="yaml_config.json"):
    """This function takes in an NxDT yaml file and converts it to a config json for checkpoint conversion.
    
    However, since we only need num_attention_heads, num_key_value_heads, num_hidden_layers, and hidden_size;
    we will just be adding those to the json as those are the only values the script needs to read. Also,
    for MoE we need num_local_experts.

    We can expand upon this in the future if needed, but this is all that's necessary right now.
    """
    yaml_data = load_yaml_file(yaml_path)
    config_json = {}
    config_json["num_hidden_layers"] = yaml_data["model"]["num_layers"]
    config_json["num_attention_heads"] = yaml_data["model"]["num_attention_heads"]
    config_json["hidden_size"] = yaml_data["model"]["hidden_size"]
    config_json["num_key_value_heads"] = yaml_data["model"]["num_kv_heads"]
    if "moe" in yaml_data["model"].keys():
        config_json["num_local_experts"] = yaml_data["model"]["moe"]["num_experts"]

    with open(filename, "w") as f:
        json.dump(config_json, f)

    return filename