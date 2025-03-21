import json
import gc  # 가비지 컬렉션 모듈

# Load original COCO dataset
with open('./coco/annotations/instances_train2017_entire.json', 'r') as f:
    data = json.load(f)
print(f"Entire object images: {len(data['images'])}") 

images = data['images']
annotations = data['annotations']
categories = data['categories']

# Define area thresholds
LARGE_THRESHOLD = 96 * 96
SMALL_THRESHOLD = 32 * 32

# Split annotations by object size
large_annotations = []
medium_annotations = []
small_annotations = []

for ann in annotations:
    area = ann['area']
    if area > LARGE_THRESHOLD:
        large_annotations.append(ann)
    elif area < SMALL_THRESHOLD:
        small_annotations.append(ann)
    else:
        medium_annotations.append(ann)

# 메모리 해제
del annotations
gc.collect()

def create_coco_subset(annotations_subset):
    image_ids = set(ann['image_id'] for ann in annotations_subset)
    images_subset = [img for img in images if img['id'] in image_ids]
    return {
        "images": images_subset,
        "annotations": annotations_subset,
        "categories": categories
    }

# Create and save datasets one by one to 절약 메모리
def save_subset(filename, annotations_subset):
    dataset = create_coco_subset(annotations_subset)
    with open(filename, 'w') as f:
        json.dump(dataset, f)
    print(f"{filename} saved: {len(dataset['images'])} images, {len(dataset['annotations'])} annotations")
    del dataset
    gc.collect()

save_subset('coco_large.json', large_annotations)
del large_annotations
gc.collect()

save_subset('coco_medium.json', medium_annotations)
del medium_annotations
gc.collect()

save_subset('coco_small.json', small_annotations)
del small_annotations
gc.collect()

# 마무리 정리
del images, categories, data
gc.collect()
