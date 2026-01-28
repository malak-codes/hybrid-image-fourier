from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def convert_image_to_grayscale(image):
    # Convert both images to grayscale
    image_grayscale = image.convert('L')
    image_array = np.array(image_grayscale)
    return image_array

def resize_by_taking_every_kth_pixel(image_array, factor=2):
    """
    Resizes the image by taking every k-th pixel (downsampling).

    Parameters:
        image_array (np.array): Input image as a NumPy array.
        factor (int): Downsampling factor (e.g., 2 for every 2nd pixel).

    Returns:
        np.array: Downsampled image array.
    """
    return image_array[::factor, ::factor]


class CloseUpFarDown:
    def __init__(self, close_up_image_path, far_down_image_path):
        self.close_up_image_path = close_up_image_path
        self.far_down_image_path = far_down_image_path
        self.close_up_image = Image.open(self.close_up_image_path)
        self.far_down_image = Image.open(self.far_down_image_path)

    def apply_fourier_transform(self, image_array):
        # Apply Fourier Transform and shift the zero frequency to the center
        dft = np.fft.fft2(image_array)
        dft_shifted = np.fft.fftshift(dft)
        magnitude_spectrum = np.log(1 + np.abs(dft_shifted))
        return dft_shifted, magnitude_spectrum

    def create_gaussian_filters(self, image_shape, cutoff_frequency):
        # Create frequency distance grid
        rows, cols = image_shape
        crow, ccol = rows // 2, cols // 2  # Center coordinates
        x = np.arange(0, cols)
        y = np.arange(0, rows)
        x, y = np.meshgrid(x, y)
        distance_matrix = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

        # Gaussian filters
        GLPF = np.exp(-(distance_matrix ** 2) / (2 * (cutoff_frequency ** 2)))  # Low-pass
        GHPF = 1 - GLPF  # High-pass (1 - Low-pass)
        return GHPF, GLPF

    def combine_fourier_transforms(self, dft_close_up, dft_far_down, GHPF, GLPF):
        """
        Combines the Fourier Transforms of the close-up and far-down images after applying filters.
        """
        # Apply filters
        high_pass_filtered = dft_close_up * GHPF
        low_pass_filtered = dft_far_down * GLPF

        # Combine filtered Fourier Transforms
        combined_fourier = high_pass_filtered + low_pass_filtered

        # Normalize combined Fourier for better scaling
        combined_fourier /= np.max(np.abs(combined_fourier))

        return combined_fourier, high_pass_filtered, low_pass_filtered

    def get_hybrid_image(self, cutoff_frequency=10):
        # Convert images to grayscale
        close_up_arr = convert_image_to_grayscale(self.close_up_image)
        far_down_arr = convert_image_to_grayscale(self.far_down_image)
        # Step 1: Fourier Transform
        dft_shifted_close_up, magnitude_close_up = self.apply_fourier_transform(close_up_arr)
        dft_shifted_far_down, magnitude_far_down = self.apply_fourier_transform(far_down_arr)

        # Step 2: Create Gaussian High-Pass and Low-Pass Filters
        GHPF_close, GLPF_close = self.create_gaussian_filters(close_up_arr.shape, cutoff_frequency)
        GHPF_far , GLPF_far = self.create_gaussian_filters(far_down_arr.shape, cutoff_frequency)
        # Step 3: Combine Fourier Transforms
        combined_fourier, high_pass_filtered, low_pass_filtered = self.combine_fourier_transforms(
            dft_shifted_close_up, dft_shifted_far_down, GHPF_close, GLPF_far
        )

        high = np.abs(np.fft.ifft2(np.fft.ifftshift(high_pass_filtered)))
        low = np.abs(np.fft.ifft2(np.fft.ifftshift(low_pass_filtered)))
        # plt.imshow(high+low, cmap='gray')
        # Step 4: Inverse Fourier Transform
        hybrid_image = np.abs(np.fft.ifft2(np.fft.ifftshift(combined_fourier)))
        # plt.imshow(hybrid_image)
        # Display intermediate results
        plt.figure(figsize=(15, 10))

        plt.subplot(3, 3, 1)
        plt.title("Close-Up Image (Grayscale)")
        plt.imshow(close_up_arr, cmap='gray')

        plt.subplot(3, 3, 2)
        plt.title("Far-Down Image (Grayscale)")
        plt.imshow(far_down_arr, cmap='gray')

        plt.subplot(3, 3, 3)
        plt.title("Fourier Transform (Close-Up)")
        plt.imshow(magnitude_close_up, cmap='gray')

        plt.subplot(3, 3, 4)
        plt.title("Fourier Transform (Far-Down)")
        plt.imshow(magnitude_far_down, cmap='gray')

        plt.subplot(3, 3, 5)
        plt.title("Gaussian High-Pass Filter")
        plt.imshow(GHPF_close, cmap='gray')

        plt.subplot(3, 3, 6)
        plt.title("Gaussian Low-Pass Filter")
        plt.imshow(GLPF_far, cmap='gray')

        plt.subplot(3, 3, 7)
        plt.title("High-Pass Filtered Fourier (Close-Up)")
        plt.imshow(np.log(1 + np.abs(high_pass_filtered)), cmap='gray')

        plt.subplot(3, 3, 8)
        plt.title("Low-Pass Filtered Fourier (Far-Down)")
        plt.imshow(np.log(1 + np.abs(low_pass_filtered)), cmap='gray')

        plt.subplot(3, 3, 9)
        plt.title("Hybrid Image")
        plt.imshow(hybrid_image, cmap='gray')

        plt.tight_layout()
        plt.show()

        return hybrid_image

########################################################################################################################
########################################################################################################################
########################################################################################################################


class SeamlessBlend:
    def __init__(self, image1_path, image2_path, binary_mask):
        self.image1 = Image.open(image1_path)
        self.image2 = Image.open(image2_path)
        if self.image1.size != self.image2.size:
            print("Images are of different sizes.")
            return
        self.image1_arr = convert_image_to_grayscale(self.image1)
        self.image2_arr = convert_image_to_grayscale(self.image2)
        self.image1_gaussian_pyramid = self.construct_gaussian_pyramid(self.image1_arr, np.log(self.image1_arr.shape[0]))
        self.image2_gaussian_pyramid = self.construct_gaussian_pyramid(self.image2_arr, np.log(self.image2_arr.shape[0]))
        self.binary_mask_gaussian_pyramid = self.construct_gaussian_pyramid(binary_mask((self.image1_arr.shape[0], self.image1_arr.shape[1])), self.image1_arr.shape[0])
        self.image1_laplacian_pyramid = self.construct_laplacian_pyramid(self.image1_gaussian_pyramid)
        self.image2_laplacian_pyramid = self.construct_laplacian_pyramid(self.image2_gaussian_pyramid)
        blended_laplacians = self.blend_pyramids(self.image1_laplacian_pyramid, self.image2_laplacian_pyramid, self.binary_mask_gaussian_pyramid)
        blended_image_arr = self.reconstruct_from_pyramid(blended_laplacians)
        self.blended_image = Image.fromarray(blended_image_arr)

    def get_blended_image(self):
        self.show_pyramids()
        return self.blended_image

    def show_pyramids(self):

        plt.figure(figsize=(15, 10))

        plt.subplot(3, 3, 1)
        plt.title("Gaussian Pyramid Level 0")
        plt.imshow(self.image2_gaussian_pyramid[0], cmap='gray')

        plt.subplot(3, 3, 2)
        plt.title("Gaussian Pyramid Level 2")
        plt.imshow(self.image2_gaussian_pyramid[2], cmap='gray')

        plt.subplot(3, 3, 3)
        plt.title("Gaussian Pyramid Level 4")
        plt.imshow(self.image2_gaussian_pyramid[4], cmap='gray')

        plt.subplot(3, 3, 4)
        plt.title("Laplacian Pyramid Level 0")
        plt.imshow(self.image2_laplacian_pyramid[0], cmap='gray')

        plt.subplot(3, 3, 5)
        plt.title("Laplacian Pyramid Level 2")
        plt.imshow(self.image2_laplacian_pyramid[2], cmap='gray')

        plt.subplot(3, 3, 6)
        plt.title("Laplacian Pyramid Level 4")
        plt.imshow(self.image2_laplacian_pyramid[4], cmap='gray')

        plt.tight_layout()
        plt.show()



    def apply_gaussian_smoothing(self, image_array):

        # Define a 3x3 Gaussian kernel
        gaussian_kernel = np.array([[1, 2, 1],
                                     [2, 4, 2],
                                     [1, 2, 1]], dtype=np.float32)
        gaussian_kernel /= np.sum(gaussian_kernel)  # Normalize the kernel

        # Perform convolution with the Gaussian kernel
        smoothed_image = self.convolve2d(image_array, gaussian_kernel)

        return smoothed_image

    def convolve2d(self,image_array, kernel):
        """
        Perform 2D convolution on an image using a kernel.

        Parameters:
            image (np.array): Input image.
            kernel (np.array): Convolution kernel.

        Returns:
            np.array: Convolved image.
        """
        kernel_height, kernel_width = kernel.shape
        image_height, image_width = image_array.shape

        # Pad the image with zeros to handle edges
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
        padded_image = np.pad(image_array, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')

        # Perform convolution
        convolved_image = np.zeros_like(image_array, dtype=np.float32)
        for i in range(image_height):
            for j in range(image_width):
                region = padded_image[i:i + kernel_height, j:j + kernel_width]
                convolved_image[i, j] = np.sum(region * kernel)

        convolved_image = np.clip(convolved_image, 0, 255)
        return convolved_image

    def construct_gaussian_pyramid(self, image_array, levels):
        """Constructs a Gaussian pyramid for the given image."""
        gaussian_pyramid = [image_array]
        for _ in range(int(levels) - 1):
            smoothed_level = self.apply_gaussian_smoothing(gaussian_pyramid[-1])
            resized_level = resize_by_taking_every_kth_pixel(smoothed_level)
            gaussian_pyramid.append(resized_level)
        return gaussian_pyramid

    def resize_with_bilinear_interpolation(self,image_array, target_shape):
        """
        Resizes an image to the target shape using bilinear interpolation.

        Parameters:
            image_array (np.array): Input image as a NumPy array.
            target_shape (tuple): Target shape as (height, width).

        Returns:
            np.array: Resized image.
        """
        # Convert NumPy array to PIL image
        image = Image.fromarray(image_array)

        # Resize using bilinear interpolation
        resized_image = image.resize(target_shape[::-1], resample=Image.BILINEAR)

        # Convert back to NumPy array
        return np.array(resized_image)

    def construct_laplacian_pyramid(self, gaussian_pyramid):
        """Constructs a Laplacian pyramid from a Gaussian pyramid."""
        laplacian_pyramid = []
        for i in range(len(gaussian_pyramid) - 1):
            # Get the target shape as (height, width) from the current Gaussian level
            target_shape = (gaussian_pyramid[i].shape[0], gaussian_pyramid[i].shape[1])
            next_level = self.resize_with_bilinear_interpolation(gaussian_pyramid[i + 1], target_shape)
            laplacian = np.array(gaussian_pyramid[i], dtype=np.float32) - np.array(next_level, dtype=np.float32)
            laplacian_pyramid.append(laplacian)
        laplacian_pyramid.append(np.array(gaussian_pyramid[-1], dtype=np.float32))  # Add the smallest level
        return laplacian_pyramid

    def blend_pyramids(self,L_a, L_b, G_m):
        """Blends two Laplacian pyramids using a Gaussian mask pyramid."""
        blended_pyramid = []
        for L_a_k, L_b_k, G_m_k in zip(L_a, L_b, G_m):
            # Ensure the mask is normalized
            if np.max(G_m_k) > 1:
                G_m_k = G_m_k / 255.0
            blended = G_m_k * L_a_k + (1 - G_m_k) * L_b_k
            blended_pyramid.append(blended)
        return blended_pyramid

    def reconstruct_from_pyramid(self, laplacian_pyramid):
        # Start with the smallest level of the Laplacian pyramid
        image = laplacian_pyramid[-1]
        for level in reversed(laplacian_pyramid[:-1]):
            # Resize the current image to match the shape of the next level
            image = self.resize_with_bilinear_interpolation(image, level.shape)

            # Add the current Laplacian level to reconstruct the image
            image = np.array(image, dtype=np.float32) + level

        # Clip the final reconstructed image to valid range and return as uint8
        return np.clip(image, 0, 255).astype(np.uint8)


# def create_gradient_mask(size, direction="horizontal"):
#     """
#     Creates a gradient mask with smooth transitions.
#     """
#     width, height = size
#     mask = Image.new("L", size, 0)
#     for i in range(width if direction == "horizontal" else height):
#         value = int(255 * i / (width if direction == "horizontal" else height))
#         if direction == "horizontal":
#             mask.paste(value, (i, 0, i + 1, height))
#         else:
#             mask.paste(value, (0, i, width, i + 1))
#     return np.array(mask)

def create_half_mask(size, direction="vertical", gradient_width=50):
    """
    Creates a mask with half the image for blending with a smooth gradient.

    Parameters:
        size (tuple): The size of the mask (width, height).
        direction (str): The direction of the division ("vertical" or "horizontal").
        gradient_width (int): The width of the gradient transition.

    Returns:
        np.array: A half-mask image with a gradient transition.
    """
    width, height = size
    mask = np.zeros((height, width), dtype=np.float32)

    if direction == "vertical":
        # Left half white, right half black
        mask[:, :width // 2 - gradient_width // 2] = 1.0  # Left side fully white
        mask[:, width // 2 - gradient_width // 2: width // 2 + gradient_width // 2] = np.linspace(
            1.0, 0.0, gradient_width
        )  # Smooth gradient
    elif direction == "horizontal":
        # Top half white, bottom half black
        mask[:height // 2 - gradient_width // 2, :] = 1.0  # Top side fully white
        mask[height // 2 - gradient_width // 2: height // 2 + gradient_width // 2, :] = np.linspace(
            1.0, 0.0, gradient_width
        ).reshape(-1, 1)  # Smooth gradient

    return mask

# def create_radial_gradient_mask(size, center=None, radius=None):
#     """
#     Creates a radial gradient mask with smooth transitions.
#
#     Parameters:
#         size (tuple): The size of the mask (width, height).
#         center (tuple): The center of the radial gradient. Defaults to the center of the image.
#         radius (int): The radius of the gradient. Defaults to half the smaller dimension.
#
#     Returns:
#         np.array: A radial gradient mask with values between 0 and 255.
#     """
#     width, height = size
#     if center is None:
#         center = (width // 2, height // 2)
#     if radius is None:
#         radius = min(width, height) // 2

#     mask = np.zeros((height, width), dtype=np.float32)
#
#     for x in range(width):
#         for y in range(height):
#             # Compute the distance from the center
#             distance = ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5
#             # Scale the distance to create a gradient
#             value = 1 - min(distance / radius, 1)
#             mask[y, x] = value
#
#     # Normalize to 0-255
#     mask = (mask * 255).astype(np.uint8)
#     return mask

if __name__ == "__main__":

    hybrid = CloseUpFarDown("wh.jpg", 'man.jpg')
    hybrid.get_hybrid_image()
    #
    blender1 = SeamlessBlend("wh.jpg", "man.jpg", create_half_mask)
    # blender2 = SeamlessBlend("wh.jpg", "man.jpg", create_gradient_mask)

    blended_image1= blender1.get_blended_image()
    # blended_image2 = blender2.get_blended_image()

    # Save and display the result
    blended_image1.save("blended_image.jpg")
    blended_image1.show()


