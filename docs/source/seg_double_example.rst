.. _seg_double_example:

Segmentation Double Example
===========================

This example demonstrates the use of double segmentation in image processing. Double segmentation is a technique used to refine the segmentation results by applying a secondary segmentation process on the initial segmented output.

Example Code
------------

Below is an example code snippet that illustrates double segmentation:

.. code-block:: python

    import cv2
    import numpy as np

    # Load the image
    image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

    # Initial segmentation
    _, initial_segmentation = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Secondary segmentation
    kernel = np.ones((5, 5), np.uint8)
    refined_segmentation = cv2.morphologyEx(initial_segmentation, cv2.MORPH_CLOSE, kernel)

    # Display results
    cv2.imshow('Initial Segmentation', initial_segmentation)
    cv2.imshow('Refined Segmentation', refined_segmentation)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

Explanation
-----------

1. **Initial Segmentation**: The image is thresholded to create a binary segmentation.
2. **Secondary Segmentation**: Morphological operations are applied to refine the segmentation results.

Applications
------------

Double segmentation is useful in scenarios where the initial segmentation is not sufficient, such as noisy images or complex structures. It helps in improving the accuracy and quality of the segmented output.
