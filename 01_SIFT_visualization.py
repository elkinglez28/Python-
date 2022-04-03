import cv2
import numpy as np
import time

# Read image and convert to grayscale
imageName = 'a0004.jpg'
ima=cv2.imread(imageName)
gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)

# Create detector and descriptor structures to compute SIFT
detector=cv2.FeatureDetector_create('SIFT')
descriptor = cv2.DescriptorExtractor_create('SIFT')

# Detect keypoints with SIFT and sort them according to their response
print 'Extracting Keypoints'
init=time.time()
kpts=detector.detect(gray)
kpts = sorted(kpts, key = lambda x:x.response)

end=time.time()
print 'Extracted '+str(len(kpts))+' keypoints.'
print 'Done in '+str(end-init)+' secs.'
print ''

# Compute SIFT descriptor for all keypoints
print 'Computing SIFT descriptors'
init=time.time()
kpts,des=descriptor.compute(gray,kpts)
end=time.time()
print 'Done in '+str(end-init)+' secs.'

# Show result of detecting keypoints
im_with_keypoints = cv2.drawKeypoints(ima, kpts, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey()


