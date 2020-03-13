#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import os
import random
from PIL import Image
from os.path import join, getsize
import argparse


# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument("--file_path", action='store')
parser.add_argument("--output_path", action='store')
parser.add_argument("--img_size", type=int)
args = parser.parse_args()


# In[ ]:


file_path = args.file_path
output_path = args.output_path
img_size = args.img_size

output_pathn = os.path.join(output_path, 'train.tfrecord')

# In[ ]:


size = 0
total = 0
for root, dirs, files in os.walk(file_path):
    size += sum([getsize(join(root, name)) for name in files])
    total += len(files)
num_shards = int(size/1024/1024/200)+1
total_div = int(total/num_shards) + 1


# In[ ]:


idir = os.listdir(file_path)
filename = []
labell = []
for imgd in idir:
    if os.path.isdir(file_path + imgd) and imgd != '.ipynb_checkpoints':
        imgl = os.listdir(file_path + imgd)
        labell.append(imgd)
        for imgp in imgl:
            filename.append(file_path + imgd + '/' + imgp)

print('total images: ' + str(len(filename)))
print('total classes: ' + str(len(labell)))
print('total tfrecords: ' + str(num_shards))

for imgclass in labell:
    labelcls = labell.index(imgclass)
    with open('./class_index.txt', 'a') as f:
        f.write(str(imgclass) + ' ' + str(labelcls) + '\n')

# In[ ]:


for num in range(num_shards):
    
    output_pathd = output_pathn + '_' + '{:05d}-of-{:05d}'.format(num, num_shards)
    writer = tf.python_io.TFRecordWriter(output_pathd)
    random.shuffle(filename)
    for i, index in enumerate(filename[total_div*num:total_div*(num+1)]):   
        
        if img_size == None:
            with tf.gfile.GFile(index, 'rb') as fid:
                encoded_jpg = fid.read()
        else:
            img = Image.open(index)
            img = img.resize((img_size, img_size),Image.ANTIALIAS)
            img.save('./tmp.png')
            with tf.gfile.GFile('./tmp.png', 'rb') as fid:
                encoded_jpg = fid.read()

    #     labeli = index.split('/')[4].split('_')[0] + '0' + index.split('/')[4].split('_')[1]
        labeli = labell.index(index.split('/')[-2])

        label = int(labeli)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
            'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[index.split('/')[-1].split('.')[0].encode('utf8')])),
            'dirname': tf.train.Feature(bytes_list=tf.train.BytesList(value=[index.split('/')[-2].encode('utf8')])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))

        writer.write(example.SerializeToString())  # Serialize To String
        
        ii = i+num*total_div+1
        if ii % 100 == 0:
            print('%d num image processed' % ii)

    writer.close()
    
if os.path.isfile('./tmp.png'):
    os.remove('./tmp.png')
    
print('%d num image processed' % ii)
print('finish')

