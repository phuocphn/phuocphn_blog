---
layout: post
title:  "Download ImageNet LSVRC 2012 Dataset"
date:   2021-11-26 11:10:04 +0900
categories: ml
tags: imagenet dataset
author: Phuoc. Pham
comments: true
---

ImageNet LSVRC 2012 is a popular dataset for ML/ Computer Vision research. A lot of papers used this dataset as standard benchmark for performance evaluation. Unfortunately, at the time of writing this article, you cannot download Imagenet dataset as it is not available to freely download online anymore. So you will have to manually download it.

There are several forums/websites provided the URLs for downloading, but most of them are also not available. The following tutorial shows how to obtain the Imagenet dataset by using Torrent.

### **ImageNet Torrent Download**
First, you need to download the torrent files of ImageNet the the following URLs.


- Training Set: [ImageNet LSVRC 2012 Training Set (Object Detection)](https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2)
- Validation Set: [ImageNet LSVRC 2012 Validation Set (Object Detection)](https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5)

and then use any Torrent client to download all the images and annotation files.

### **Directory Construction**

After the download is complete, you have to create a standard directory that contains the `train` and `val` sub directories (which is required for DL frameworks to understand). Please refer to the following bash script in order to construct it in a correct way.

```bash
# Unpacking the training dataset
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar
rm -f ILSVRC2012_img_train.tar #(If you want to delete the original compressed file)
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

# Unpacking the Validation dataset
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

```


### **Crosscheck**
You should verify the number of image files to make sure that everything is correctly loaded. If the number of image files is the same as the bellow number, then your installation is correct.

```bash
cd ~/dataset/ILSVRC2012
find train/ -name "*.JPEG" | wc -l
#1281167
find val/ -name "*.JPEG" | wc -l
#50000
```



#### **References:**
1. [ImageNet LSVRC 2012 데이터셋 다운로드 받기](https://seongkyun.github.io/others/2019/03/06/imagenet_dn/)
2. [ImageNet下载地址](https://simon32.github.io/2018/01/09/image-net/)
3. [soumith/imagenet-multiGPU.torch](https://github.com/soumith/imagenet-multiGPU.torch)
4. [Working with ImageNet (ILSVRC2012) Dataset in NVIDIA DIGITS](https://jkjung-avt.github.io/ilsvrc2012-in-digits/)


{% include disqus_comments.html %}