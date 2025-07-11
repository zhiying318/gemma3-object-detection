from datasets import load_dataset, DownloadConfig
import time

 # Force re-download the dataset without using the cache
# dataset = load_dataset(
#     "zhiyingzou0202/testtt",
#     split="train+test+validation",
#     # repo_type="dataset",
#     # revision="main",
#    # download_config=DownloadConfig(force_download=True)  
# )
# print("!!!",len(dataset))


def label_replace(example, keyword, new_name):
    if keyword in example["name_label"]:
        example["name_label"] = new_name
        example["bbox_location_name_label"] = example["bbox_location_name_label"][:37] + new_name
    return example

def remove_brackets(example):
    if example["name_label"].startswith("[") and example["name_label"].endswith("]"):
        example["name_label"] = example["name_label"][2:-2]
    return example



splits = ["train", "test", "validation"]
for split in splits:
    dataset = load_dataset("zhiyingzou0202/object_detection_bbox_3", split=split)
    print("LABELS EXIST:", set(dataset["name_label"]))
    print("!!!", split, len(dataset))
    time.sleep(2)
    dataset_updated = dataset.map(remove_brackets)
    dataset_updated = dataset_updated.map(label_replace, fn_kwargs={"keyword": "paper plate", "new_name": "shredded purple cabbage"})
    dataset_updated = dataset_updated.map(label_replace, fn_kwargs={"keyword": "items", "new_name": "shredded purple cabbage"})
    dataset_updated = dataset_updated.map(label_replace, fn_kwargs={"keyword": "stacked sticks", "new_name": "shredded purple cabbage"})
    dataset_updated = dataset_updated.map(label_replace, fn_kwargs={"keyword": "shredded carrot", "new_name": "shredded carrots"})
    dataset_updated.push_to_hub(
        "zhiyingzou0202/object_detection_bbox_3",
        split=split,
        create_pr=False,
        commit_message=f"label_replace for {split}, after adding shredded purple cabbage",
    )