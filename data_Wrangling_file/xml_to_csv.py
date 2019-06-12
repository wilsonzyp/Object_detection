# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 00:52:02 2018
@author: Xiang Guo
changed by wilson 20190610 1830
将该文件夹下train、test文件夹内的xml文件内的信息统一记录到.csv表格中（xml_to_csv.py）。
"""
import glob
import pandas as pd
import xml.etree.ElementTree as ET

DIRS = {'train': r'train', 'test': r'test'}


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):  # 找到所有标注的块
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


if __name__ == '__main__':
    for name, image_path in DIRS.items():
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(name + '.csv', index=None)
        print('Successfully converted ' + image_path + r'/*.xml to ' + name + '.csv')
