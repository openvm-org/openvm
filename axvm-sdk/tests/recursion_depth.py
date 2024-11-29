import json

def calculate_json_depth(data, current_depth=0):
    if isinstance(data, dict):
        return max(
            (calculate_json_depth(value, current_depth + 1) for value in data.values()),
            default=current_depth
        )
    elif isinstance(data, list):
        return max(
            (calculate_json_depth(item, current_depth + 1) for item in data),
            default=current_depth
        )
    else:
        return current_depth

def main():
    file_path = "/tmp/axvm-sdk-it/agg_pk.json"
    with open(file_path, "r") as file:
        data = json.load(file)

    for item in data:
        depth = calculate_json_depth(item)
        print(f"Maximum recursion depth: {depth}")

if __name__ == "__main__":
    main()