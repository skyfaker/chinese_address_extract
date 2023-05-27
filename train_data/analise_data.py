import os

file_path_list = ['./train.txt', './dev.txt']
clear_label_pattern = dict()
for file_path in file_path_list:
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    print("analise file from {}".format(file_path))
    with open(file_path, "r", encoding='utf-8') as f:
        line_list = f.readlines()

    gt_label_list = []
    for line in line_list:
        line = line.strip()
        if line != '':
            text, label = line.split()
            if label == "O":
                continue
            label = label.split('-')[1]
            if label in ["prov", "city", "district", "subpoi"]:
                continue
            if gt_label_list != [] and label == gt_label_list[-1]:
                continue
            gt_label_list.append(label)
        else:
            if gt_label_list:
                pattern = '-'.join(gt_label_list)
                if pattern in clear_label_pattern:
                    clear_label_pattern[pattern] += 1
                else:
                    clear_label_pattern[pattern] = 1
                gt_label_list = []

sort_dict = sorted(clear_label_pattern.items(), key=lambda kv: (kv[1], kv[0]))
t = 0
c = 0
for k, v in sort_dict:
    c += 1
    print("{}: {}".format(k, v))
    t += v
print("total: {}, kind: {}".format(t, c))
