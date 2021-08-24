#!/usr/bin/env python3

import os
import sys
import shutil

if len(sys.argv) != 3:
    print(f'Usage: {sys.argv[0]} <src_dir> <tar_dir>')
    os.exit(-1)

src_dir = sys.argv[1]
tar_dir = sys.argv[2]

for img in os.listdir(src_dir):
    ori_path = os.path.join(src_dir, img)
    new_path = os.path.join(tar_dir, '/'.join(img.split('_')))
    new_dir = os.path.dirname(new_path)
    os.makedirs(new_dir, exist_ok=True)

    shutil.copy(ori_path, new_path)

