# Cancer_Image_Classfication_Capstone
Lung And Colon Cancer Image Classfication
About the dataset Original Article Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. Lung and Colon Cancer Histopathological Image Dataset (LC25000). arXiv:1912.12142v1 [eess.IV], 2019

Relevant Links https://arxiv.org/abs/1912.12142v1 https://github.com/tampapath/lung_colon_image_set

Dataset BibTeX @article{, title= {LC25000 Lung and colon histopathological image dataset}, keywords= {cancer,histopathology}, author= {Andrew A. Borkowski, Marilyn M. Bui, L. Brannon Thomas, Catherine P. Wilson, Lauren A. DeLand, Stephen M. Mastorides}, url= {https://github.com/tampapath/lung_colon_image_set} }

This dataset contains 25,000 histopathological images with 5 classes. All images are 768 x 768 pixels in size and are in jpeg file format. The images were generated from an original sample of HIPAA compliant and validated sources, consisting of 750 total images of lung tissue (250 benign lung tissue, 250 lung adenocarcinomas, and 250 lung squamous cell carcinomas) and 500 total images of colon tissue (250 benign colon tissue and 250 colon adenocarcinomas) and augmented to 25,000 using the Augmentor package. There are five classes in the dataset, each with 5,000 images, being:

1) Lung benign tissue

2) Lung adenocarcinoma

3) Lung squamous cell carcinoma

4) Colon adenocarcinoma

5) Colon benign tissue

Introduction:
Lung and colon cancer is currently one of the most vital diseases in society, and patients are more likely to be cured if the disease is spotted earlier. Using computer vision for analyzing the lung and colon cancer images will spead up the analysis process.

Computer vision is a field of computer science that works on enabling computers to see, identify and process images in the same way that human vision does, and then provide appropriate output. It is like imparting human intelligence and instincts to a computer. Computer vision comes from modelling image processing using the techniques of machine learning, computer vision applies machine learning to recognize patterns for interpretation of images (much like the process of visual reasoning of human vision).

I will be using the technique of transfer learning to implement a neural network for lung cancer image classfication

The Principle of Transfer Learning
Transfer learning is a machine learning method where a model developed for a certain task is reused as the starting point for other tasks.

In the field of Deep Learning, this technique is usually the method of which pre-trained models are used as the starting point on computer vision and natural language processing tasks.

By using the method of Transfer Learning, we can save the total training time for the model, a better performance of neural networks (in most cases), and requires less data.
