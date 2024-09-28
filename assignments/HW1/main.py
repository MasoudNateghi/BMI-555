import imageio
import numpy as np
from skimage.draw import line
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import radon, iradon

#%% Q1
# Read the phantom image
img = plt.imread('assignments/HW1/phantom.gif')
plt.imshow(img, cmap='gray')
plt.savefig('assignments/HW1/plots/phantom.png', format='png')
plt.show()
plt.close()

# Q1-part-a
img_fft = np.fft.fft2(img)  # 2D Fourier transform
img_fft_shift = np.fft.fftshift(img_fft)  # shift the zero frequency component to the center
img_mag = np.abs(img_fft_shift)  # magnitude of the Fourier transform
img_phase = np.angle(img_fft_shift)  # phase of the Fourier transform

# plot log magnitude of the Fourier transform
plt.figure()
plt.imshow(np.log(img_mag), cmap='gray')
plt.savefig('assignments/HW1/plots/Q1_part_a.png', format='png')
plt.show()
plt.close()

# Q1-part-b
img_fft_inv = np.fft.ifft2(img_fft)  # inverse Fourier transform
img_inv = np.abs(img_fft_inv)  # magnitude of the inverse Fourier transform

# plot the inverse Fourier transform
plt.figure()
plt.imshow(img_inv, cmap='gray')
plt.savefig('assignments/HW1/plots/Q1_part_b.png', format='png')
plt.show()
plt.close()

# Q1-part-c
img_fft_loss = img_fft_shift.copy()  # copy the FFT of the image
m, n = img_fft_loss.shape  # size of the FFT
patch_size = 10  # size of the patch

# define the coordinates of the patch in the FFT
row_start = (m - patch_size) // 2
row_end = (m + patch_size) // 2
col_start = (n - patch_size) // 2
col_end = (n + patch_size) // 2

img_fft_loss[row_start:row_end, col_start:col_end] = 0  # remove the patch from the FFT
img_fft_loss_inv = np.fft.fft2(img_fft_loss)  # inverse Fourier transform
img_loss = np.abs(img_fft_loss_inv)  # magnitude of the inverse Fourier transform

# plot the image with the removed patch
plt.figure()
plt.imshow(img_loss, cmap='gray')
plt.savefig('assignments/HW1/plots/Q1_part_c.png', format='png')
plt.show()
plt.close()

# Q1-part-d
img_fft_center = np.zeros_like(img_fft_shift)  # create a zero matrix with the same size as the FFT
img_fft_center[row_start:row_end, col_start:col_end] = img_fft_shift[row_start:row_end, col_start:col_end]  # keep the patch in the center of the FFT
img_fft_center_inv = np.fft.ifft2(img_fft_center)  # inverse Fourier transform
img_center = np.abs(img_fft_center_inv)  # magnitude of the inverse Fourier transform

# plot the image with only the patch in the center of the FFT
plt.figure()
plt.imshow(img_center, cmap='gray')
plt.savefig('assignments/HW1/plots/Q1_part_d.png', format='png')
plt.show()
plt.close()

# Q1-part-e
'''
When taking inverse Fourier transform of the image, the components in the center of the FFT are associated with the 
lower frequency components of the image. Therefore, when we remove the components in the center of the FFT, we are
removing the low frequency components of the image. As a result, we could only see the sharp transitions in the image, 
which are called the edges of the image and are associated with high frequency components of the image. On the other 
hand, when we keep only the components in the center of the FFT, we are keeping the low frequency components of the 
image. As a result, we could only see the smooth transitions in the image, which results in a blurry image. 
'''


#%% Q2-part-a
angles = [0, 45, 90]  # projection angles
sinogram = radon(img, theta=angles, circle=True)  # radon transform

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(sinogram[:, 0])
plt.ylabel('Intensity (0°)')

plt.subplot(3, 1, 2)
plt.plot(sinogram[:, 1])
plt.ylabel('Intensity (45°)')

plt.subplot(3, 1, 3)
plt.plot(sinogram[:, 2])
plt.ylabel('Intensity (90°)')
plt.xlabel('Projection position')

plt.savefig('assignments/HW1/plots/Q2_part_a.png', format='png')
plt.show()
plt.close()

# Q2-part-b
angles = np.arange(0, -361, -1)  # projection angles
sinogram = radon(img, theta=angles)  # radon transform

# Q2-part-c
img_iradon = iradon(sinogram, theta=angles, filter_name=None)  # inverse radon transform without filtering
plt.imshow(img_iradon, cmap='gray')
plt.savefig('assignments/HW1/plots/Q2_part_c.png', format='png')
plt.show()
plt.close()

# Q2-part-d
img_iradon_filtered = iradon(sinogram, theta=angles, filter_name='ramp')  # inverse radon transform with ramp filter
plt.imshow(img_iradon_filtered, cmap='gray')
plt.savefig('assignments/HW1/plots/Q2_part_d.png', format='png')
plt.show()
plt.close()

# Q2-part-e
'''
Reconstructed image using ramp filter looks very similar to the original image. Filtered image has less artifacts and
is sharper than the image reconstructed without any filter. This is because the ramp filter is a high-pass filter
that removes the low frequency components of the image. As a result, the image reconstructed with the ramp filter has
sharper edges compared to the image reconstructed without any filter. The Reconstructed image without any filter is
much more smoother and more blurry compared to the image reconstructed with the ramp filter and is almost useless.
'''


#%% Q3
img_norm = img / np.max(img)  # normalize the image between 0 and 1
img_noisy = img_norm + 0.1 * np.random.randn(*img_norm.shape)  # add Gaussian noise to the image with N(0, 0.1)
img_noisy = np.clip(img_noisy, 0, 1)  # clip the noisy image between 0 and 1

# plot the noisy image
plt.figure()
plt.imshow(img_noisy, cmap='gray')
plt.savefig('assignments/HW1/plots/Q3_noisy.png', format='png')
plt.show()
plt.close()

# Q3-part-a
angles = [0, 45, 90]  # projection angles
sinogram_noisy = radon(img_noisy, theta=angles, circle=True)  # radon transform of the noisy image

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(sinogram_noisy[:, 0])
plt.ylabel('Intensity (0°)')

plt.subplot(3, 1, 2)
plt.plot(sinogram_noisy[:, 1])
plt.ylabel('Intensity (45°)')

plt.subplot(3, 1, 3)
plt.plot(sinogram_noisy[:, 2])
plt.ylabel('Intensity (90°)')
plt.xlabel('Projection position')

plt.savefig('assignments/HW1/plots/Q3_part_a.png', format='png')
plt.show()
plt.close()

# Q3-part-b
angles = np.arange(0, -361, -1)  # projection angles
sinogram = radon(img_noisy, theta=angles)  # radon transform of the noisy image

# Q3-part-c
img_iradon_noisy = iradon(sinogram, theta=angles, filter_name=None)  # inverse radon transform without filtering
plt.imshow(img_iradon_noisy, cmap='gray')
plt.savefig('assignments/HW1/plots/Q3_part_c.png', format='png')
plt.show()
plt.close()

# Q3-part-d
img_iradon_filtered_noisy = iradon(sinogram, theta=angles, filter_name='ramp')  # inverse radon transform with ramp filter
plt.imshow(img_iradon_filtered_noisy, cmap='gray')
plt.savefig('assignments/HW1/plots/Q3_part_d.png', format='png')
plt.show()
plt.close()

# Q3-part-e
'''
Adding noise to the image results in a noisy sinogram, which leads to a noisy reconstructed image in both filtered and 
unfiltered cases. Specially the ringing artifacts are more pronounced in the noisy reconstructed image compared to the
original image. The ringing artifacts are caused by the high frequency components in the image (noise), which are amplified 
by the reconstruction process. However, the image reconstructed with no filter is blurrier compared to the image
reconstructed with the ramp filter and is almost useless. 
'''


#%% Q4
# Read the gif file
gif_3ch = imageio.mimread('assignments/HW1/Ultrasound_of_human_heart_apical_4-cahmber_view.gif') * 20 # repeated the gif ten times
gif = [rgb2gray(frame) for frame in gif_3ch]  # convert to grayscale

# Define the start and end points of the line for M-mode
start_point = (15, 160)
end_point = (200, 200)
rr, cc = line(start_point[0], start_point[1], end_point[0], end_point[1])

# Extract the M-mode data
m_mode = np.array([frame[rr, cc] for frame in gif]).T

# Plot the M-mode data
plt.figure()
plt.imshow(m_mode, cmap='gray')
plt.xlabel('Time')
plt.ylabel('Depth')
plt.savefig('assignments/HW1/plots/Q4.png', format='png')
plt.show()
plt.close()


#%% Q5
# unit circle coordinates
x_cir = np.linspace(0, 1, 1000)
y_cir = np.sqrt(1 - x_cir ** 2)

# Generate random uniform points in [0, 1] interval for x and y
nPoints = 1000
x = np.random.uniform(size=nPoints)
y = np.random.uniform(size=nPoints)

# Calculate the number of points inside the circle
n_inside = np.sum(x ** 2 + y ** 2 < 1)
pi_approx = 4 * n_inside / x.size
print(f'Approximation of pi: {pi_approx}')

# plot the circle and the random points
plt.figure()
plt.plot(x_cir, y_cir, 'r')
plt.scatter(x, y, s=4)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Approximation of pi: {pi_approx}')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('assignments/HW1/plots/Q5.png', format='png')
plt.show()
plt.close()
