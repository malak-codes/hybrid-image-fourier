# Hybrid Image Generation: Fourier Transform & Gaussian Filters

This repository contains an implementation of **Hybrid Image Generation** using frequency domain analysis. The project explores how to combine high-frequency details from one image with low-frequency components of another to create an image that changes based on viewing distance.

## Project Overview

The core objective of this project is to implement the "Close-Up Far-Down" effect. By applying a Gaussian High-Pass Filter (GHPF) to a close-up image and a Gaussian Low-Pass Filter (GLPF) to a far-down image, we can merge them in the frequency domain using the Fast Fourier Transform (FFT).

### Key Features

- **Fourier Transform Analysis**: Uses `numpy.fft` to transform images into the frequency domain for filtering.
- **Gaussian Filters**: Implements custom Gaussian High-Pass and Low-Pass filters with adjustable cutoff frequencies.
- **Hybrid Image Synthesis**: Combines filtered frequency components and performs an Inverse Fourier Transform to reconstruct the hybrid image.
- **Visualization**: Includes scripts to visualize the magnitude spectrum and intermediate filtering results.

## Repository Structure

| File | Description |
| :--- | :--- |
| `ex3.py` | The main implementation script containing the `CloseUpFarDown` class and filtering logic. |
| `man.jpg` | Sample source image for high-frequency details. |
| `wh.jpg` | Sample source image for low-frequency components. |
| `README.md` | Documentation providing an overview of the project and its components. |
| `LICENSE` | MIT License for the project. |

## Requirements

The project requires the following Python libraries:
- `numpy`
- `matplotlib`
- `Pillow`

You can install the dependencies using:
```bash
pip install numpy matplotlib Pillow
```

## Usage

To run the hybrid image generation:
```bash
python ex3.py
```
The script will process the sample images and display the resulting hybrid image along with its frequency spectrum.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## Authors
- Malak Laham
- Zenab Waked
