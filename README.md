## FID-FID : Fake Image Detector meets Frequency - Image Dual domain

<!--
<p align="center">
    <img src="assets/NAT-Diffuser-logo.png" alt="Alt text" width="300">
</p>
<p align="center">
    <img src="assets/NAT-faces.png" alt="Alt text" width="800">
</p>
-->

<div align="center">
    2024.04.24 ~ 2024.06.14
</div>


<div align="center">
    <a href="">Report(coming soon)</a> | 
    <a href="">PPT(coming soon)</a>
</div>



<br>

<p align="center">
    <img src="FID-Fid.jpg" alt="Alt text" width="650">
</p>


**FID-FID** is the **F**ake **I**mage **D**etector based on **F**requency and **I**mage **D**ual domain. We made feature extractor as dual branch, one for image-domain and other for frequency domain. Both branch are based on ViT encoder part, we can extract both image-domain features and frequecy-domain features better than CNN based extractor. After passing the dual branch module, we apply cross-attention with two branches outputs and pass it to binary classification module. Here, from cross-attention module, we can apply both image and frequency domain information to the binary classification.


## Setup



### Requirements

Here are the required libraries list. By using the code bellow, you can easily install all required libraries

```
pip install -r requirements.txt
```




### Dataset Setup

For the **Real** images, we used dataset from [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces?select=train.csv) for Deepfake Detection Challenge in Kaggle. Note there are 50k real iamges among 140k. For the **Fake** images, we used dataset from [NAT-Diffuser](https://github.com/justin4ai/NAT-Diffuser), project for fake image generation part linked to this fake image detection part. For the **Test** dataset, from [Chicago Face Database](https://www.chicagofaces.org/) and [Synthetic Faces High Quality(SFHQ)](https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-1), we used 250 images for each real and fake image.
The whole dataset we used looks like bellow.
```
datasets
┣ test
┃ ┣ 001.jpg
┃ ┃ ┣ ...
┃ ┣ 500.jpg
┃ ┗ test_labels.csv
┗ train
  ┣ generated
  ┃ ┣ samples_0.png
  ┃ ┣ ...
  ┃ ┗ samples_999.png
  ┗ real
    ┣ 00000.jpg
    ┣ ...
    ┗ 69999.jpg
```


### Pre-Trained Models
On the side of Image-domain, we used pre-trained model for feature extraction from [HuggingFace Transformer](https://github.com/huggingface/transformers/tree/v4.41.2), there are several various feature extractor models that can be used instead of what we exactly used.
Also, the Pre-trained models of our entire model can be found [here](). 


## Commands

### Training
|Arguments|option(default)|
|------|------|
|--real_folder_name|Path of the folder where the real images for train has stored ("./datasets/train/real")|
|--fake_folder_name|Path of the folder where the fake images for train has stored ("./datasets/train/generated")|
|--test_folder_name|Path of the folder where the test images has stored ("./datasets/test")|
|--num_epochs|Number of epochs to train (50)|
|--batch_size|Mini batch Size (16)|
|--use_checkpoint|Rather to use pre-trained checkpoint of model (False(default) : Not, Ture : use)|
|--save_path|Path of the folder where the model checkpoints has stored, <br> or where you want to save the checkpoints ("./checkpoints/")|
|--learning_rate|Learning rate of the optimizer (0.005)|
|--weight_decay|Weight decay of the optimizer (0.005)|

You can simply train the model using code bellow. To apply traing to specific environments, you can use those arguments above to change some options while training.   

```python train.py```

### Evaluation

|Arguments|option(default)|
|------|------|
|--test_folder_name|Path of the folder where the test images has stored ("./datasets/test")|
|--batch_size|Mini batch Size (16)|
|--checkpoint_path|Path of the folder where the model checkpoints has stored ("./checkpoints/")|
|--checkpoint|Which iteration of checkpoint your going to use (50)|

As same as training session, you can simply test the model using code bellow. Also you can apply specific environments by using arguments above to change some options while testing.  
```python evaluation.py```

<!--
### Web Demo
-->


## References

The code for this project refer from [Huggingface/transformer](https://github.com/huggingface/transformers/tree/v4.41.2/src/transformers) for building image-domain ViT block, [F3Net](https://github.com/yyk-wew/F3Net) and [FreqNet-DeepfakeDetection](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection) for building frequency-domain ViT block. 



## Citation
      

