# Image Translation Comparisons

This is the final project for "Deep Learning：Fundamentals and Applications" at National Chengchi University (NCCU) by Liang, Shih-Rui, Hsiao, Hung-Hsuan, and Hung, Jui-Fu. Full report is [here](https://drive.google.com/file/d/1AZO57pdeNEYIapIESyMtDnGuQD4yrzeJ/view?usp=share_link).

# Abstract

There are several ways to generate new images based on provided samples. Well-known examples include Cycle Generative Adversarial Network (CycleGAN) and Adaptive Convolutions (AdaConv). Previous studies [1, 2] show that the two models have excellent image-to-image translation results. In this study, the two models are fed with different image datasets and are then used to turn real-world images into respective datasets' styles. The generator of CycleGAN is implemented with both ResNet [3] and U-Net [4]. Sobel filters [5] are added on some training data to test the edge detection effects. The results show that for different tasks, certain combinations of training data, models, or model structures outperform others.

# Full Experiment Structure

CycleGAN, CycleGAN with Unet as generator, and AdaConv are the three models adopted in this research. The models are trained with Chinese Ink, Monet, and Ukiyo-e paintings and can then be used to turn real-world images (domain X) into respective styles (domain Y).

![](https://static.coderbridge.com/img/juifuhung/217a25f842c44155a1ec7f99b9a62c7a.png)

The original images and style images are downloaded from Kaggle. Styles of the images include Chinese Landscape Painting Dataset (referred as Chinese Ink in this study), Monet2Photo (referred as Monet in this study), and Ukiyo-e2photo (referred as Ukiyo-e in this study). Sobel filter is applied to training data of the original images, creating images with obvious white object edges. The three sets are combined in pairs each time they are sent to the models for training.

#### Sources of Training Data:

- [Chinese Ink](https://www.kaggle.com/datasets/myzhang1029/chinese-landscape-painting-dataset)
- [ Monet](https://www.kaggle.com/datasets/balraj98/monet2photo?fbclid=IwAR0QIn37ieLg9ysAR3WZ8XA0A3MsQohKH_ZSh7cSGVD5Tk-AHjN1BDBIk7I)
- [ Ukiyo-e](https://www.kaggle.com/datasets/helloeyes/ukiyoe2photo?fbclid=IwAR3cq1u5dtkj25tvEuhMqQpvt-XueSFvBFvX0LUuGIgOa5f6qQKLCxXTVVw)

#### Models Adopted:

- CycleGAN (with ResNet)
- CycleGAN (with U-Net)
- AdaConv

# Results and Discussions

#### Chinese Ink

![](https://static.coderbridge.com/img/juifuhung/76a40c6db9124e1ea44d8c5157eed513.png)
The batch size is 1, number of epochs is 100, and the learning rate is 10⁻⁵. Adam is set as the optimizer. AdaConv has the best results. All 9 output images in this group are similar to training data and have equally good qualities. CycleGAN with ResNet also outputs Chinese Ink-style images, but are not as stable. CycleGAN with U-Net does little on image translation. The images are still colorful, losing the expected greyish styles of Chinese Ink.

#### Monet

![](https://static.coderbridge.com/img/juifuhung/66f798b1e0224c3080f5541c265103ea.png)
The batch size is 1, number of epochs is 100, and the learning rate is 10⁻⁵. Adam is set as the optimizer. CycleGAN with ResNet performs the best, with the most obvious Impressionism styles. The brush strokes are bold, and more emphasis is placed on lights. CycleGAN with U-Net also translates images to similar styles, but its results are not as good as the ones produced by CycleGAN with ResNet. Output images of AdaConv have the worst results. The colors in all images in this group are similar, which do not show the iconic features of the training data.

#### Ukiyo-e

![](https://static.coderbridge.com/img/juifuhung/88960f4f057548ea8bc48f13da893589.png)
Output images are generally poor. The results from CycleGAN with both ResNet and U-Net preserve the characteristic of color patches from Ukiyo-e, but the patches spread randomly across images. AdaConv results, however, lose such characteristics. The colors are harmonious, which are very different from a typical Ukiyo-e-style image.

# Conclusions

This research compares the image translation results of different models, variations of a model, and different combinations of training data. The model plays the most crucial role, as changes of models make the most differences. There is no model that outperforms all other models in all kinds of tasks. CycleGAN with ResNet is better at Monet-style translation; AdaConv is better at Chinese Ink-style translation; CycleGAN with both ResNet and U-net as generators are equally good at Ukiyo-e-style translation. For edge detection, CycleGAN with U-Net has the least effect. It indicates that an adequate model should be chosen depending on the mission. Using the wrong model will lead to poor results given the same training data and parameters.

# References

1. J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision, 2017.
1. P. Chandran, G. Zoss, P. Gotardo, M. Gross, D. Bradley. Adaptive Convolutions for Structure-Aware Style Transfer. CVPR46437.2021.00788, 2021.
1. K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016, IEEE Computer Society, 2016, pp. 770–778.
1. O. Ronneberger, P. Fischer, T. Brox. U-net: Convolutional networks for biomedical image segmentation, in: International Conference on Medical image computing and computer-assisted intervention, Springer, 2015, pp. 234–241.
1. N. Kanopoulos, N. Vasanthavada, R. L. Baker. Design of an Image Edge Detection Filter Using the Sobel Operator, IEEE Journal of Solid-State Circuits, vol. 23, no. 2, 1988.
