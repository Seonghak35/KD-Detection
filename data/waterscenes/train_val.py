import json
import os

# === 파일 경로 설정 ===
coco_ann_path = '/HDD/guest/ssg/KD-Detection/data/waterscenes/annotations/annotations.json'  # 전체 annotation JSON
train_txt = '/HDD/guest/ssg/KD-Detection/data/waterscenes/train.txt'
val_txt = '/HDD/guest/ssg/KD-Detection/data/waterscenes/val.txt'
output_dir = '/HDD/guest/ssg/KD-Detection/data/waterscenes/annotations/'  # 저장할 위치
os.makedirs(output_dir, exist_ok=True)

# === 1. 파일 이름 추출 ===
def load_img_ids(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    return set([os.path.basename(line.strip()) for line in lines])

train_imgs = load_img_ids(train_txt)
val_imgs = load_img_ids(val_txt)

# === 2. 원본 COCO JSON 불러오기 ===
with open(coco_ann_path, 'r') as f:
    coco = json.load(f)

# === 3. 이미지 ID 기준으로 분할 ===
def split_coco(coco, target_imgs, name=''):
    new_coco = {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'categories': coco['categories'],
        'images': [],
        'annotations': []
    }

    valid_img_ids = set()

    for img in coco['images']:
        if img['file_name'] in target_imgs:
            new_coco['images'].append(img)
            valid_img_ids.add(img['id'])

    for ann in coco['annotations']:
        if ann['image_id'] in valid_img_ids:
            new_coco['annotations'].append(ann)

    print(f"[{name}] Images: {len(new_coco['images'])}, Annotations: {len(new_coco['annotations'])}")
    return new_coco

# === 4. 분할된 JSON 저장 및 개수 출력 ===
train_coco = split_coco(coco, train_imgs, name='Train')
val_coco = split_coco(coco, val_imgs, name='Val')

with open(os.path.join(output_dir, 'instances_train.json'), 'w') as f:
    json.dump(train_coco, f)

with open(os.path.join(output_dir, 'instances_val.json'), 'w') as f:
    json.dump(val_coco, f)
