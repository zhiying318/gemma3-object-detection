from datasets import load_dataset

dataset = load_dataset(
    "zhiyingzou0202/object_detection_bbox_paligemma",
    #,  # 上一步找到的 commit hash 或 tag 名
)

train_split = dataset["train"]
test_split = dataset["test"]

# # 随机切分出 10%  / 10% 测试
# splits = test_split.train_test_split(test_size=0.2, seed=42)

# 将 test 再分成 50% val, 50% test（即各占总数据集 10%）
test_val_split = test_split.train_test_split(test_size=0.5, seed=42)
val_split = test_val_split["train"]   # 10%
final_test_split = test_val_split["test"]  # 10%

# 重新组织 splits
final_splits = {
    "train": train_split,
    "validation": val_split,
    "test": final_test_split,
}
print(f"Train split size: {len(train_split)}", 
      f"Validation split size: {len(val_split)}", 
      f"Test split size: {len(final_test_split)}")

# splits.push_to_hub(
#     "zhiyingzou0202/object_detection_bbox_paligemma",
#     create_pr=False,    
#     commit_message="Add train/test/validation splits to last version of dataset"
# )

from datasets import DatasetDict
DatasetDict(final_splits).push_to_hub(
    "zhiyingzou0202/object_detection_bbox_paligemma",
    create_pr=False,
    commit_message="Add validation split (10%) and adjust test split (10%)"
)