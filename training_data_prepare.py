import json
import os

img_path_list = []
txt_list = []
text_img_pairs = []

root_dir = 'datasets/laion_aesthetics_6.5'
for file in os.listdir(root_dir):
    file = os.path.join(root_dir, file)
    if os.path.isdir(file):
        for subfile in os.listdir(file):
            file_name = subfile.split('.')[0]
            file_suffix = subfile.split('.')[1]
            if file_suffix in ['jpg', 'png', 'jpeg']:
                img_path = os.path.join(file, subfile)
                text_path = os.path.join(file, file_name + '.txt')
                img_path_list.append(img_path)
                with open(text_path, 'rt') as f:
                    line = f.readline()
                    txt_list.append(line)


for img_path, txt in zip(img_path_list, txt_list):
    text_img_pairs.append({'img_path': img_path, 'prompt': txt})


with open('datasets/training_data.json', 'a') as f:
    for i in range(len(text_img_pairs)):
        json.dump(text_img_pairs[i], f)
        if i != len(text_img_pairs) - 1:
            f.write('\n')
