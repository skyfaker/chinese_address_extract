import random


def prov_city_district_augment():
    prov_city, city_district = {' ': ["北京市", "天津市", "上海市", "重庆市"]}, {}
    with open('./China-district.txt', encoding='utf-8') as f:
        lines = f.readlines()
        curr_prov, curr_city = None, None
        for line in lines:
            code, name = line.strip().split('\t')
            name = name.strip()
            int_code = int(code)
            if int_code % 10000 == 0 and name[-1] != "市":
                curr_prov = name
                if name not in prov_city:
                    prov_city[name] = []
            elif int_code % 100 == 0:
                curr_city = name
                if curr_prov:
                    prov_city[curr_prov].append(name)
                if name not in city_district:
                    city_district[name] = []
            else:
                city_district[curr_city].append(name)

    prov_city_district = []
    # for prov in prov_city:
    #     for city in prov_city[prov]:
    #         for district in city_district[city]:
    #             prov_name = prov.strip()
    #             # if prov.strip() and prov[-1] == "省" and random.random() < 0.5:
    #             #     prov_name = prov[:-1]
    #             city_name = city
    #             if city.strip() and city[-1] == "市" and random.random() < 0.5:
    #                 city_name = city[:-1]
    #
    #             if len(district) > 2 and district[-1] in ["县", "市", "区"]:
    #                 flag = True
    #                 for n in ["矿区", "自治县"]:
    #                     if n in district:
    #                         flag = False
    #                         break
    #                 if flag:
    #                     prov_city_district.append([prov_name, city_name, district[:-1]])
    #             prov_city_district.append([prov_name, city_name, district])

    tmp_prov = ["北京市", "天津市", "上海市", "重庆市"]
    for prov in prov_city:
        for city in prov_city[prov]:
            for district in city_district[city]:
                city_name = city
                if city.strip() and city not in tmp_prov and city[-1] == "市" and random.random() < 0.5:
                    city_name = city[:-1]

                if len(district) > 2 and district[-1] in ["县", "市", "区"]:
                    flag = True
                    for n in ["矿区", "自治县"]:
                        if n in district:
                            flag = False
                            break
                    if flag:
                        prov_city_district.append(['', city_name, district[:-1]])
                prov_city_district.append(['', city_name, district])

    with open('./prov-city-district.txt', 'w', encoding='utf-8') as f:
        for prov_city_district_item in prov_city_district:
            seq = ''.join(prov_city_district_item)
            label_list = []
            p_name, c_name, d_name = prov_city_district_item[0], prov_city_district_item[1], prov_city_district_item[2]
            if p_name:
                tmp = ["I-prov"] * len(p_name)
                tmp[0] = "B-prov"
                tmp[-1] = "E-prov"
                label_list.extend(tmp)
            if c_name:
                tmp = ["I-city"] * len(c_name)
                tmp[0] = "B-city"
                tmp[-1] = "E-city"
                label_list.extend(tmp)
            if d_name:
                tmp = ["I-district"] * len(d_name)
                tmp[0] = "B-district"
                tmp[-1] = "E-district"
                label_list.extend(tmp)

            if len(seq) == len(label_list):
                for s, l in zip(seq, label_list):
                    f.write(s + " " + l + "\n")
                f.write("\n")
    print('finish')


def train_data_augment():
    with open('/train.txt', encoding='utf-8') as f:
        lines = f.readlines()

        element_text = ''
        label_type = None
        all_element_dict = {}
        for line in lines:
            line = line.strip()
            if line != '':
                text, label = line.split()
                if label == "O":
                    continue
                if label.split('-')[0] == 'B':
                    label_type = label.split('-')[1]
                    element_text = ''
                    element_text += text.strip()
                elif label.split('-')[0] == 'E':
                    element_text += text.strip()
                    if label_type == 'poi':
                        if element_text[-1] not in ['里'] or len(element_text) <= 2:
                            label_type = None
                            element_text = ''
                            continue
                        else:
                            print(element_text, end=', ')
                    if label_type not in all_element_dict:
                        all_element_dict[label_type] = []
                    if element_text not in all_element_dict[label_type]:
                        all_element_dict[label_type].append(element_text)
                    label_type = None
                else:
                    element_text += text.strip()

    all_element_dict['poi'].extend(["小关北里", "潘家园南里", '十里堡北里', '西坝河西里', '百子湾西里', '百子湾东里', '和平里',
                                    '安贞里', '六里屯西里', '大井东里', '永华南里', '甜水园东里', '红莲南里', '望园东里'])
    augment_type = []
    with open('./augment_data_type.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line != '':
                augment_type.append(line)

    new_data = []
    new_label = []
    for aug_type in augment_type:
        for i in range(500):
            for t in aug_type.split('-'):
                tmp_text = all_element_dict[t][random.randint(0, len(all_element_dict[t])-1)]
                tmp_label = ['I-' + t] * len(tmp_text)
                tmp_label[0] = 'B-' + t
                tmp_label[-1] = 'E-' + t
                new_data.extend(list(tmp_text))
                new_label.extend(tmp_label)
            new_data.append('')
            new_label.append('')

    with open('./aug_train.txt', 'w', encoding='utf-8') as f:
        for t, l in zip(new_data, new_label):
            f.write(t + " " + l + "\n")
    print('\n finish')


if __name__ == '__main__':
    # prov_city_district_augment()
    train_data_augment()
