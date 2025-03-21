import json

# Load original COCO dataset
with open('./tiny-coco/annotations/instances_train2017.json', 'r') as f:
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

def create_coco_subset(annotations_subset):
    image_ids = set(ann['image_id'] for ann in annotations_subset)
    images_subset = [img for img in images if img['id'] in image_ids]
    return {
        "images": images_subset,
        "annotations": annotations_subset,
        "categories": categories
    }

# Create datasets
large_dataset = create_coco_subset(large_annotations)
medium_dataset = create_coco_subset(medium_annotations)
small_dataset = create_coco_subset(small_annotations)

# Save to JSON files
with open('coco_large.json', 'w') as f:
    json.dump(large_dataset, f)
with open('coco_medium.json', 'w') as f:
    json.dump(medium_dataset, f)
with open('coco_small.json', 'w') as f:
    json.dump(small_dataset, f)

# Print number of unique images in each subset
print(f"Large object images: {len(large_dataset['images'])}")
print(f"Medium object images: {len(medium_dataset['images'])}")
print(f"Small object images: {len(small_dataset['images'])}")
