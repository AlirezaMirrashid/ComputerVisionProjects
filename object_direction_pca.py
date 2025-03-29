import cv2
import numpy as np

# Step 1: Preprocessing the Image
image = cv2.imread('samples/samp1.jpg')  # Read the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
edges = cv2.Canny(gray, 100, 200)  # Edge detection using Canny

# Step 2: Contour Detection (or directly find points of interest)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Assuming we're interested in the largest contour
contour = max(contours, key=cv2.contourArea)

# Step 3: Perform PCA on the Contour Points
# Flatten the contour points into a 2D array (X, Y coordinates)
points = contour.reshape(-1, 2)

# Step 4: Center the points (mean subtraction)
mean = np.mean(points, axis=0)
centered_points = points - mean

# Step 5: PCA to find the principal components (directions)
cov_matrix = np.cov(centered_points.T)  # Covariance matrix
eigvals, eigvecs = np.linalg.eig(cov_matrix)  # Eigenvalues and eigenvectors

# Sort eigenvalues and eigenvectors in descending order of eigenvalue
sorted_indices = np.argsort(eigvals)[::-1]  # Indices of eigenvalues sorted in descending order
eigvals_sorted = eigvals[sorted_indices]
eigvecs_sorted = eigvecs[:, sorted_indices]

# The two eigenvectors corresponding to the largest eigenvalues
principal_direction_1 = eigvecs_sorted[:, 0]  # First principal direction
principal_direction_2 = eigvecs_sorted[:, 1]  # Second principal direction

# Step 6: Draw the Two Directions on the Image
# Get the center of the object (mean of points)
center = tuple(mean.astype(int))

# Scale the direction vectors for visualization
length = 100  # Length of the arrows

# End points for the first and second directions
end_point_1 = (int(center[0] + principal_direction_1[0] * length),
               int(center[1] + principal_direction_1[1] * length))

end_point_2 = (int(center[0] + principal_direction_2[0] * length),
               int(center[1] + principal_direction_2[1] * length))

# Draw the arrows on the image
cv2.arrowedLine(image, center, end_point_1, (0, 255, 0), 2)  # First principal direction
cv2.arrowedLine(image, center, end_point_2, (255, 0, 0), 2)  # Second principal direction

# Step 7: Show the Image with the Two Directions
cv2.imshow('Image with Two Directions', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
