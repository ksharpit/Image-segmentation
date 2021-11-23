
import numpy as np
from matplotlib import pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM

img = cv2.imread("flo.jpg")
plt.imshow(img)
# Convert MxNx3 image into Kx3 where K=MxN
img2 = img.reshape((-1, 3))  #-1 reshape means, in this case MxN

n=input('Enter number of clusters\n')
print('Enter\n1 for k-means clustering\n2 for GMM')
cmd=int(input('enter the desired image segmentation algorithm\n'))


if cmd == 1:
    #for K Mean cluster
    kmeans = KMeans(n=2, init='k-means++', max_iter=300, n_init=10, random_state=42)
    model = kmeans.fit(img2)
    predicted_values = kmeans.predict(img2)

    #res = center[label.flatten()]
    segm_image = predicted_values.reshape((img.shape[0], img.shape[1]))
    plt.imshow(segm_image)
    #plt.imshow(segm_image, cmap='gray')
    segm_image = np.expand_dims(segm_image, axis=-1)
    
    foreground = np.multiply(segm_image, img)
    background = img - foreground
    plt.imshow(foreground) 
    cv2.imwrite('fore.jpg', foreground)
    plt.imshow(background)
    cv2.imwrite('back.jpg', background)
    
    
elif cmd ==2:
    #for GMM cluster
    #covariance choices, full, tied, diag, spherical
    gmm_model = GMM(n_components=2, covariance_type='tied').fit(img2)  #tied works better than full
    gmm_labels = gmm_model.predict(img2)
    
    #Put numbers back to original shape so we can reconstruct segmented image
    original_shape = img.shape
    segmented = gmm_labels.reshape(original_shape[0], original_shape[1])
    segmented = np.expand_dims(segmented, axis=-1)
    plt.imshow(segmented)
    
    foreground = np.multiply(segmented, img)
    
    background = img - foreground
    plt.imshow(foreground) 
    cv2.imwrite('fore.jpg', foreground)
    plt.imshow(background)
    cv2.imwrite('back.jpg', background)

else:
    print('Invalid input')
    



