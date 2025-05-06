"""
Advanced scratch detection algorithms for semiconductor wafers.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.ndimage import label, binary_dilation
from skimage.measure import regionprops
from skimage.morphology import skeletonize


def detect_scratch_dies_advanced(wafer_df, model, wafer_yield_threshold=0.8):
    """
    Advanced approach to predict which dies are part of a scratch based on the wafer map

    Args:
        wafer_df: DataFrame containing dies from a single wafer
        model: Trained model for scratch detection
        wafer_yield_threshold: Threshold for low yield wafers

    Returns:
        wafer_df: Original DataFrame with an additional column for scratch predictions
    """
    result_df = wafer_df.copy()
    result_df['IsScratchDie_Pred'] = 0

    # Check if this is a low yield wafer
    wafer_yield = wafer_df['IsGoodDie'].mean()
    if wafer_yield <= wafer_yield_threshold:
        return result_df  # No scratch detection for low yield wafers

    # Generate wafer map
    wafer_map = generate_wafer_map(wafer_df)
    wafer_map_rgb = np.repeat(wafer_map.reshape(1, 52, 52, 1), 3, axis=3)

    # Predict if the wafer has a scratch
    has_scratch_prob = model.predict(wafer_map_rgb)[0][0]
    has_scratch = has_scratch_prob >= 0.5

    if has_scratch:
        # Extract bad die coordinates
        bad_dies = wafer_df[wafer_df['IsGoodDie'] == 0]
        if len(bad_dies) <= 3:  # Not enough bad dies to form a scratch
            return result_df

        # Convert to binary map for processing
        binary_map = np.zeros((52, 52), dtype=np.uint8)
        for _, die in bad_dies.iterrows():
            x, y = int(die['DieX']), int(die['DieY'])
            if 0 <= x < 52 and 0 <= y < 52:
                binary_map[y, x] = 1

        # Approach 1: DBSCAN Clustering for detecting groups
        coords = np.array([[die['DieX'], die['DieY']] for _, die in bad_dies.iterrows()])
        dbscan = DBSCAN(eps=2.5, min_samples=3)  # Dies within 2.5 units are considered neighbors
        clusters = dbscan.fit_predict(coords)

        # Approach 2: Linear pattern detection
        scratch_coords = set()

        # Get the properties of clusters
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Skip noise points
                continue

            # Get the coordinates for this cluster
            cluster_coords = coords[clusters == cluster_id]

            # If the cluster has few points, skip
            if len(cluster_coords) < 3:
                continue

            # Check if the cluster forms a line-like pattern
            linearity_score = calculate_linearity(cluster_coords)

            if linearity_score < 0.3:  # Lower score means more linear
                # Add all points from this cluster to scratch coordinates
                for x, y in cluster_coords:
                    scratch_coords.add((x, y))
            elif len(cluster_coords) >= 8:  # Large clusters may be scratches even if not perfectly linear
                for x, y in cluster_coords:
                    scratch_coords.add((x, y))

        # Approach 3: Skeletonization to find connected paths
        # Dilate first to connect nearby points
        dilated = binary_dilation(binary_map, structure=np.ones((3, 3)))

        # Skeletonize to find the "backbone" of potential scratches
        skeleton = skeletonize(dilated)

        # Label connected components in the skeleton
        labeled, num_features = label(skeleton)

        # Examine each skeleton component
        for i in range(1, num_features + 1):
            component = (labeled == i)
            props = regionprops(component.astype(int))[0]

            # Scratches tend to be elongated
            if props.area > 3 and props.eccentricity > 0.7:
                y_coords, x_coords = np.where(component)
                for x, y in zip(x_coords, y_coords):
                    # Find actual die coordinates near this skeleton point
                    for _, die in bad_dies.iterrows():
                        die_x, die_y = int(die['DieX']), int(die['DieY'])
                        if abs(die_x - x) <= 1 and abs(die_y - y) <= 1:
                            scratch_coords.add((die_x, die_y))

        # Approach 4: Edge detection in neighborhood
        edge_coords = detect_edges_in_neighborhood(binary_map)
        for x, y in edge_coords:
            for _, die in bad_dies.iterrows():
                die_x, die_y = int(die['DieX']), int(die['DieY'])
                if die_x == x and die_y == y:
                    scratch_coords.add((die_x, die_y))

        # Mark dies as scratch
        for i, row in wafer_df.iterrows():
            if (row['DieX'], row['DieY']) in scratch_coords:
                result_df.loc[i, 'IsScratchDie_Pred'] = 1

    return result_df


def calculate_linearity(coordinates):
    """
    Calculate how linear a set of points is using PCA

    Args:
        coordinates: Array of x,y coordinates

    Returns:
        score: Linearity score (lower is more linear)
    """
    if len(coordinates) < 3:
        return 1.0

    # Calculate the covariance matrix
    cov_matrix = np.cov(coordinates.T)

    # Get eigenvalues
    eigenvalues, _ = np.linalg.eig(cov_matrix)

    # Sort eigenvalues in descending order
    eigenvalues = sorted(eigenvalues, reverse=True)

    # Calculate the ratio of the smaller eigenvalue to the larger one
    # This is a measure of how linear the points are
    # The smaller the ratio, the more linear the points
    if eigenvalues[0] == 0:
        return 0  # Perfectly linear

    return eigenvalues[1] / eigenvalues[0]


def detect_edges_in_neighborhood(binary_map, window_size=5):
    """
    Detect edge patterns in local neighborhoods

    Args:
        binary_map: Binary map of bad dies
        window_size: Size of the sliding window

    Returns:
        edge_coords: Set of coordinates that appear to be part of edges
    """
    edge_coords = set()
    height, width = binary_map.shape

    # Define directional filters for detecting lines
    horizontal_filter = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
    vertical_filter = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
    diagonal1_filter = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
    diagonal2_filter = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])

    for y in range(1, height - window_size):
        for x in range(1, width - window_size):
            # Extract local window
            window = binary_map[y - 1:y + 2, x - 1:x + 2]
            if np.sum(window) < 3:  # Not enough points
                continue

            # Apply filters
            h_score = np.sum(window * horizontal_filter)
            v_score = np.sum(window * vertical_filter)
            d1_score = np.sum(window * diagonal1_filter)
            d2_score = np.sum(window * diagonal2_filter)

            # Check if any pattern has a high score
            if max(h_score, v_score, d1_score, d2_score) > 3:
                edge_coords.add((x, y))

    return edge_coords


def generate_wafer_map(wafer_df, size=52):
    """
    Generate a wafer map image from the dataframe of a single wafer

    Args:
        wafer_df: DataFrame containing dies from a single wafer
        size: Size of the output image (size x size pixels)

    Returns:
        image: A 2D array representing the wafer map with 0s for empty spots,
               1s for good dies, 2s for bad dies, 3s for scratch dies
    """
    # Initialize empty wafer map
    wafer_map = np.zeros((size, size))

    # Fill in the wafer map with die information
    for _, die in wafer_df.iterrows():
        x, y = int(die['DieX']), int(die['DieY'])
        if x < size and y < size:  # Ensure we don't exceed the map size
            if die['IsGoodDie'] == 1:
                wafer_map[y, x] = 1  # Good die
            else:
                wafer_map[y, x] = 2  # Bad die
                if 'IsScratchDie' in die and die['IsScratchDie'] == 1:
                    wafer_map[y, x] = 3  # Scratch die

    return wafer_map