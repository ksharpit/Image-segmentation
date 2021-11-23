from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import matplotlib.pyplot as plt
import cv2


n=int(input('Enter number of clusters\n'))
print(type(n))
print('Enter\n1 for k-means clustering\n2 for GMM')
cmd=input('enter the desired image segmentation algorithm\n')

img = cv2.imread("flow.jpg")
plt.imshow(img)
# Convert MxNx3 image into Kx3 where K=MxN
img2 = img.reshape((-1, 3))  #-1 reshape means, in this case MxN


if cmd == '1':
    
    print('Initiating K-means clustering')
    #for K Mean cluster
    kmeans = KMeans(n, init='k-means++', max_iter=300, n_init=10, random_state=42)
    # k-means++ ensures that you get don’t fall into the random initialization trap.
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
    plt.imshow(background)
  
    
    
    
    
elif cmd == '2':
    print('Initiating GMM')
    #for GMM cluster
    #covariance choices, full, tied, diag, spherical
    gmm_model = GMM(n, covariance_type='tied').fit(img2)  #tied works better than full
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
    



