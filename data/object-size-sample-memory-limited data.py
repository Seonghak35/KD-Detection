import json
import gc  # 가비지 컬렉션 모듈

# Load original COCO dataset
with open('./coco/annotations/instances_train2017.json', 'r') as f:
    data = json.load(f)
print(f"Entire object images: {len(data['images'])}") 

images = data['images']
annotations = data['annotations']
categories = data['categories']

# Define area thresholds
LARGE_THRESHOLD = 96 * 96
SMALL_THRESHOLD = 32 * 32
MAX_IMAGES = 10000

# 객체 크기에 따른 image_id 수집 (annotation 기준)
image_to_area = {}  # image_id: [annotation areas]

for ann in annotations:
    img_id = ann['image_id']
    area = ann['area']
    if img_id not in image_to_area:
        image_to_area[img_id] = []
    image_to_area[img_id].append(area)

# 이미지 분류
large_img_ids = []
medium_img_ids = []
small_img_ids = []

for img_id, areas in image_to_area.items():
    for area in areas:
        if area > LARGE_THRESHOLD:
            large_img_ids.append(img_id)
            break
        elif area < SMALL_THRESHOLD:
            small_img_ids.append(img_id)
            break
        else:
            medium_img_ids.append(img_id)
            break
        
def limit_ids(id_list):
    return set(id_list[:MAX_IMAGES])

large_img_ids = limit_ids(list(dict.fromkeys(large_img_ids)))
medium_img_ids = limit_ids(list(dict.fromkeys(medium_img_ids)))
small_img_ids = limit_ids(list(dict.fromkeys(small_img_ids)))


# subset 생성 함수
def create_subset(image_ids, name):
    subset_images = [img for img in images if img['id'] in image_ids]
    subset_annotations = [ann for ann in annotations if ann['image_id'] in image_ids]
    subset = {
        "images": subset_images,
        "annotations": subset_annotations,
        "categories": categories
    }
    with open(f"coco_{name}.json", 'w') as f:
        json.dump(subset, f)
    print(f"coco_{name}.json saved: {len(subset_images)} images, {len(subset_annotations)} annotations")
    gc.collect()

# 저장
create_subset(large_img_ids, "large")
create_subset(medium_img_ids, "medium")
create_subset(small_img_ids, "small")
