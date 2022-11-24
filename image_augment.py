import imageio.v2 as imageio
import imgaug as ia
import imgaug.augmenters as iaa
import ipyplot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


img = mpimg.imread('screenshot.jpg')
# print(img)
imgplot = plt.imshow(img)

input_img = imageio.imread('screenshot.jpg')

rot1 = iaa.Affine(rotate=(-50,20))
input_rot1 = rot1.augment_image(input_img)
images_list=[input_img, input_rot1]
labels = ['Original', 'Rotated Image']
ipyplot.plot_images(images_list,labels=labels,img_width=180)