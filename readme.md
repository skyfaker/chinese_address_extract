## 介绍
使用bert+biafine模型和bert+bi-lstm+crf模型抽取中文地址中的实体（具体类别见[train_data/中文地址要素解析标注规范.pdf](https://github.com/skyfaker/chinese_address_extract/blob/master/train_data/%E4%B8%AD%E6%96%87%E5%9C%B0%E5%9D%80%E8%A6%81%E7%B4%A0%E8%A7%A3%E6%9E%90%E6%A0%87%E6%B3%A8%E8%A7%84%E8%8C%83.pdf)）

## 文件结构
![](/train_data/文件结构.png)

code: 代码文件
    
    main_biafine.py: 基于bert和双仿射的模型入口
    dataset: 数据类定义
    model: 模型和网络定义
    utils: 工具方法

train_data: 数据文件，包括数据分析、转换和生成代码

## 使用方法
使用bert_biaffine模型：执行`python main_biafine.py` 即可
使用bert_bi-lstm+crf模型：按照main_biafine.py文件替换成对应的网络即可