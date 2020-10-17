from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import urllib
import PIL.Image as Image
from sklearn.model_selection import train_test_split

clothing = []
with open("clothing.json") as f:
    for line in f:
        clothing.append(json.loads(line))
print(len(clothing))
print(clothing[0])

categories = []
for c in clothing:
  for a in c['annotation']:
    categories.extend(a['label'])
categories = list(set(categories))
categories.sort()
print(categories)

train_clothing, val_clothing = train_test_split(clothing, test_size=0.2)


def create_dataset(clothing, categories, dataset_type):
  images_path = Path(f"clothing/images/{dataset_type}")
  images_path.mkdir(parents=True, exist_ok=True)
  labels_path = Path(f"clothing/labels/{dataset_type}")
  labels_path.mkdir(parents=True, exist_ok=True)
  for img_id, row in enumerate(tqdm(clothing)):
    image_name = f"{img_id}.jpeg"
    img = urllib.request.urlopen(row["content"])
    img = Image.open(img)
    img = img.convert("RGB")
    img.save(str(images_path / image_name), "JPEG")
    label_name = f"{img_id}.txt"
    with (labels_path / label_name).open(mode="w") as label_file:
      for a in row['annotation']:
        for label in a['label']:
          category_idx = categories.index(label)
          points = a['points']
          p1, p2 = points
          x1, y1 = p1['x'], p1['y']
          x2, y2 = p2['x'], p2['y']
          bbox_width = x2 - x1
          bbox_height = y2 - y1
          label_file.write(
            f"{category_idx} {x1 + bbox_width / 2} {y1 + bbox_height / 2} {bbox_width} {bbox_height}\n"
          )
          

create_dataset(train_clothing, categories, 'train')
create_dataset(val_clothing, categories, 'val')