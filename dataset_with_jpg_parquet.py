from datasets import Dataset, Features, Image, Value, Sequence
import pandas as pd
import numpy as np
from PIL import Image as PILImage
import create_dataset

img = PILImage.open("./tiger.jpeg")
width, height = img.size
# width, height = int(width), int(height)

data_dict = {
    "image_id": [2],
    "image": ["tiger.jpeg"],
    "width": [width],
    "height": [height],
    "objects": [
        {"id": [1], "bbox": [[100.0, 150.0, 20.0, 25.0]], "category": ["0"]}
    ],
    "name_label": ["1_tiger"]
}
# turn data_dict into huggingface dataset
dataset = Dataset.from_dict(data_dict)
dataset = dataset.map(create_dataset.format_objects)

# format of mock data , 
features = Features({
    "image_id": Value("int64"),
    "image": Image(),
    "width": Value("int32"),
    "height": Value("int32"),
    "objects": Sequence({
        "id": Value("int64"),
        "bbox": Sequence(Value("float32"), length=4),
        "category": Value("string")
    }),
    "name_label": Value("string"),
    "bbox_location_label": Value("string")
})

dataset = dataset.cast(features)

dataset.push_to_hub("zhiyingzou0202/test4combine")