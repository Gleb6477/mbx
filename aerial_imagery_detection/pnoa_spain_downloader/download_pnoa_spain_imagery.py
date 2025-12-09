import os
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import warnings
import math
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
from pyproj import Transformer


class DoubleSolidLineDetector:
    def __init__(self, output_dir="double_solid_lines_output"):
        """
        Initialize double solid line detector for pnoa_z_x_y.png format

        Parameters:
        - output_dir: Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize coordinate transformer (Web Mercator to WGS84)
        self.transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

        # For tracking all double lines across all images
        self.all_double_lines = []  # Will store GeoJSON features

        # Detection parameters (optimized for aerial imagery)
        self.MIN_LINE_LENGTH = 40  # Minimum line length in pixels
        self.MAX_LINE_GAP = 15  # Maximum gap between line segments
        self.PARALLEL_THRESHOLD = 12  # Max angle difference for parallel lines (degrees)
        self.DISTANCE_THRESHOLD = (8, 25)  # Min and max distance between double lines (pixels)
        self.MIN_OVERLAP_RATIO = 0.6  # Minimum overlap ratio for double lines
        self.WHITE_THRESHOLD = 200  # Threshold for white detection

        # Create subdirectories for each processing step
        self.create_step_directories()

    def create_step_directories(self):
        """Create directories for each processing step"""
        self.step_dirs = {
            'original': self.output_dir / "01_original",
            'white_mask_hsv': self.output_dir / "02_white_mask_hsv",
            'white_mask_gray': self.output_dir / "02_white_mask_gray",
            'white_mask_adaptive': self.output_dir / "02_white_mask_adaptive",
            'white_mask_combined': self.output_dir / "02_white_mask_combined",
            'white_mask_cleaned': self.output_dir / "03_white_mask_cleaned",
            'edges': self.output_dir / "04_edges",
            'all_lines': self.output_dir / "05_all_lines",
            'parallel_pairs': self.output_dir / "06_parallel_pairs",
            'double_lines': self.output_dir / "07_double_lines",
            'final_binary': self.output_dir / "08_final_binary",
            'visualization': self.output_dir / "09_visualization",
            'binary_images': self.output_dir / "binary_images"  # For compatibility
        }

        for dir_path in self.step_dirs.values():
            dir_path.mkdir(exist_ok=True)

    def parse_filename(self, filename):
        """
        Parse pnoa_z_x_y.png filename format

        Expected format: pnoa_z_x_y.png
        Example: pnoa_18_123456_789012.png

        Returns:
        - Dictionary with z, x, y values and bbox
        """
        try:
            # Remove extension and split by underscores
            stem = Path(filename).stem  # pnoa_z_x_y
            parts = stem.split('_')

            if len(parts) != 4:
                print(f"Warning: Filename '{filename}' doesn't match expected format pnoa_z_x_y.png")
                return None

            # Extract values
            z = int(parts[1])  # zoom level
            x = int(parts[2])  # tile x coordinate
            y = int(parts[3])  # tile y coordinate

            # Calculate bounding box using the CORRECT method from downloader
            bbox = self.calculate_bbox_from_tile_correct(x, y, z)

            return {
                'z': z,
                'x': x,
                'y': y,
                'bbox': bbox,
                'filename': filename
            }

        except (ValueError, IndexError) as e:
            print(f"Error parsing filename '{filename}': {e}")
            return None

    def calculate_bbox_from_tile_correct(self, x, y, z):
        """
        Calculate Web Mercator bounding box from tile coordinates
        Using the SAME calculation as the downloader script

        Parameters:
        - x, y, z: Tile coordinates

        Returns:
        - bbox: [minx, miny, maxx, maxy] in Web Mercator (EPSG:3857)
        """
        # This matches the tile_to_webmercator_bbox from downloader
        # First calculate lat/lon bounds
        n = 2.0 ** z
        lon_west = x / n * 360.0 - 180.0
        lon_east = (x + 1) / n * 360.0 - 180.0

        # Latitude calculation using Mercator projection
        lat_north_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat_north = math.degrees(lat_north_rad)

        lat_south_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
        lat_south = math.degrees(lat_south_rad)

        # Convert to Web Mercator (EPSG:3857) - MATCHES DOWNLOADER
        earth_radius = 6378137.0

        # Longitude to X
        minx = lon_west * (earth_radius * math.pi / 180.0)
        maxx = lon_east * (earth_radius * math.pi / 180.0)

        # Latitude to Y (Mercator projection)
        miny = math.log(math.tan((90 + lat_south) * math.pi / 360.0)) * earth_radius
        maxy = math.log(math.tan((90 + lat_north) * math.pi / 360.0)) * earth_radius

        return [minx, miny, maxx, maxy]

    def detect_double_solid_lines(self, image, filename):
        """
        Detect double solid lines in an image and save intermediate steps

        Parameters:
        - image: Input image (BGR format)
        - filename: Base filename for saving steps

        Returns:
        - binary_image: Black background with white double lines
        - double_line_pairs: List of detected double line pairs
        """
        # Save original image
        cv2.imwrite(str(self.step_dirs['original'] / f"{filename}_original.png"), image)

        # 1. Extract white regions (road markings) and save each step
        white_mask, intermediate_masks = self.extract_white_regions_with_steps(image, filename)

        # 2. Detect all lines
        all_lines, edges = self.detect_lines_with_edges(white_mask)

        # Save edges
        cv2.imwrite(str(self.step_dirs['edges'] / f"{filename}_edges.png"), edges)

        # 3. Save all detected lines
        self.save_all_lines_visualization(image, all_lines, filename)

        # 4. Find parallel line pairs
        parallel_pairs = self.find_parallel_line_pairs(all_lines)

        # Save parallel pairs visualization
        self.save_parallel_pairs_visualization(image, parallel_pairs, filename)

        # 5. Filter for double solid lines (close, parallel, overlapping)
        double_lines = self.filter_double_lines(parallel_pairs)

        # Save double lines visualization
        self.save_double_lines_visualization(image, double_lines, filename)

        # 6. Create binary output
        binary_image = self.create_binary_output(image.shape, double_lines)

        # Save final binary image
        cv2.imwrite(str(self.step_dirs['final_binary'] / f"{filename}_final_binary.png"), binary_image)

        # 7. Create comprehensive visualization
        self.create_comprehensive_visualization(image, intermediate_masks, edges,
                                                all_lines, parallel_pairs, double_lines,
                                                binary_image, filename)

        return binary_image, double_lines

    def extract_white_regions_with_steps(self, image, filename):
        """
        Extract white regions from image and save each step
        Enhanced to reduce white/grays and smooth noise

        Parameters:
        - image: BGR image
        - filename: Base filename for saving

        Returns:
        - white_mask: Final cleaned white mask
        - intermediate_masks: Dictionary with all intermediate masks
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to smooth while preserving edges
        smoothed = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        smoothed_hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
        smoothed_gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)

        # Method 1: HSV-based white detection with dynamic threshold
        # Use value channel to dynamically determine threshold
        v_channel = hsv[:, :, 2]
        v_mean = np.mean(v_channel)
        dynamic_thresh = min(220, max(180, v_mean + 20))

        # Wider range to catch more whites and light grays
        lower_white_hsv = np.array([0, 0, dynamic_thresh])
        upper_white_hsv = np.array([180, 50, 255])
        white_mask_hsv = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)

        # Method 1b: HSV detection on smoothed image
        v_channel_smoothed = smoothed_hsv[:, :, 2]
        v_mean_smoothed = np.mean(v_channel_smoothed)
        dynamic_thresh_smoothed = min(220, max(180, v_mean_smoothed + 20))

        lower_white_hsv_smooth = np.array([0, 0, dynamic_thresh_smoothed])
        upper_white_hsv_smooth = np.array([180, 45, 255])
        white_mask_hsv_smooth = cv2.inRange(smoothed_hsv, lower_white_hsv_smooth, upper_white_hsv_smooth)

        cv2.imwrite(str(self.step_dirs['white_mask_hsv'] / f"{filename}_hsv.png"), white_mask_hsv)
        cv2.imwrite(str(self.step_dirs['white_mask_hsv'] / f"{filename}_hsv_smooth.png"), white_mask_hsv_smooth)

        # Method 2: Grayscale thresholding with adaptive reduction
        # Reduce white by using higher threshold for pure white, lower for grays
        _, white_mask_high = cv2.threshold(gray, self.WHITE_THRESHOLD + 30, 255, cv2.THRESH_BINARY)
        _, white_mask_medium = cv2.threshold(gray, self.WHITE_THRESHOLD - 10, 255, cv2.THRESH_BINARY)
        _, white_mask_low = cv2.threshold(gray, self.WHITE_THRESHOLD - 30, 255, cv2.THRESH_BINARY)

        # Combine for better gray detection
        white_mask_gray = cv2.bitwise_or(white_mask_high, white_mask_medium)
        white_mask_gray = cv2.bitwise_or(white_mask_gray, white_mask_low)

        # Grayscale on smoothed image
        _, white_mask_gray_smooth = cv2.threshold(smoothed_gray, self.WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)

        cv2.imwrite(str(self.step_dirs['white_mask_gray'] / f"{filename}_gray.png"), white_mask_gray)
        cv2.imwrite(str(self.step_dirs['white_mask_gray'] / f"{filename}_gray_smooth.png"), white_mask_gray_smooth)

        # Method 3: Adaptive thresholding - better for varying lighting
        # Gaussian adaptive threshold
        adaptive_gaussian = cv2.adaptiveThreshold(gray, 255,
                                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 31, 4)

        # Mean adaptive threshold
        adaptive_mean = cv2.adaptiveThreshold(gray, 255,
                                              cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY, 31, 4)

        adaptive_mask = cv2.bitwise_or(adaptive_gaussian, adaptive_mean)

        # Adaptive on smoothed image
        adaptive_gaussian_smooth = cv2.adaptiveThreshold(smoothed_gray, 255,
                                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv2.THRESH_BINARY, 21, 3)

        cv2.imwrite(str(self.step_dirs['white_mask_adaptive'] / f"{filename}_adaptive.png"), adaptive_mask)
        cv2.imwrite(str(self.step_dirs['white_mask_adaptive'] / f"{filename}_adaptive_smooth.png"),
                    adaptive_gaussian_smooth)

        # Method 4: Color-based white detection (RGB space)
        # Pure white has high values in all channels with low saturation
        b, g, r = cv2.split(image)

        # White areas: all channels are high and similar
        white_rgb = cv2.bitwise_and(
            cv2.threshold(r, 200, 255, cv2.THRESH_BINARY)[1],
            cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)[1]
        )
        white_rgb = cv2.bitwise_and(white_rgb,
                                    cv2.threshold(b, 200, 255, cv2.THRESH_BINARY)[1])

        # Light gray areas: all channels are moderately high and similar
        gray_rgb = cv2.bitwise_and(
            cv2.threshold(r, 180, 255, cv2.THRESH_BINARY)[1],
            cv2.threshold(g, 180, 255, cv2.THRESH_BINARY)[1]
        )
        gray_rgb = cv2.bitwise_and(gray_rgb,
                                   cv2.threshold(b, 180, 255, cv2.THRESH_BINARY)[1])

        color_mask = cv2.bitwise_or(white_rgb, gray_rgb)
        cv2.imwrite(str(self.step_dirs['white_mask_hsv'] / f"{filename}_color.png"), color_mask)

        # Combine all methods with weighted combination
        combined_mask = cv2.bitwise_or(white_mask_hsv, white_mask_hsv_smooth)
        combined_mask = cv2.bitwise_or(combined_mask, white_mask_gray)
        combined_mask = cv2.bitwise_or(combined_mask, white_mask_gray_smooth)
        combined_mask = cv2.bitwise_or(combined_mask, adaptive_mask)
        combined_mask = cv2.bitwise_or(combined_mask, adaptive_gaussian_smooth)
        combined_mask = cv2.bitwise_or(combined_mask, color_mask)

        cv2.imwrite(str(self.step_dirs['white_mask_combined'] / f"{filename}_combined.png"), combined_mask)

        # Enhanced cleaning with morphological operations
        # First, remove salt-and-pepper noise
        median_filtered = cv2.medianBlur(combined_mask, 3)

        # Morphological opening to remove small noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        opened = cv2.morphologyEx(median_filtered, cv2.MORPH_OPEN, kernel_open)

        # Morphological closing to connect nearby white regions
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        # Additional dilation to ensure lines are connected
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))  # Horizontal emphasis
        dilated = cv2.dilate(closed, kernel_dilate, iterations=1)

        # Remove small objects (noise)
        cleaned_mask = self.remove_small_objects(dilated, min_size=30)

        # Additional: Remove large blobs that are not line-like
        cleaned_mask = self.filter_by_aspect_ratio(cleaned_mask, min_aspect_ratio=2.0)

        # Smooth edges with Gaussian blur and re-threshold
        blurred = cv2.GaussianBlur(cleaned_mask, (3, 3), 0.5)
        _, cleaned_mask = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)

        cv2.imwrite(str(self.step_dirs['white_mask_cleaned'] / f"{filename}_cleaned.png"), cleaned_mask)

        # Store all intermediate masks
        intermediate_masks = {
            'hsv': white_mask_hsv,
            'hsv_smooth': white_mask_hsv_smooth,
            'gray': white_mask_gray,
            'gray_smooth': white_mask_gray_smooth,
            'adaptive': adaptive_mask,
            'adaptive_smooth': adaptive_gaussian_smooth,
            'color': color_mask,
            'combined': combined_mask,
            'cleaned': cleaned_mask,
            'smoothed_image': smoothed
        }

        return cleaned_mask, intermediate_masks

    def filter_by_aspect_ratio(self, mask, min_aspect_ratio=2.0):
        """
        Filter objects by aspect ratio to keep only line-like structures

        Parameters:
        - mask: Binary mask
        - min_aspect_ratio: Minimum aspect ratio (width/height or height/width)

        Returns:
        - filtered_mask: Mask with only line-like objects
        """
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Create output mask
        output = np.zeros_like(mask)

        for i in range(1, num_labels):  # Skip background (0)
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]

            if width == 0 or height == 0:
                continue

            # Calculate aspect ratio (long side / short side)
            aspect_ratio = max(width, height) / min(width, height)

            # Keep only objects that are line-like (high aspect ratio)
            if aspect_ratio >= min_aspect_ratio:
                output[labels == i] = 255

        return output

    def remove_small_objects(self, mask, min_size=50):
        """Remove small disconnected objects from mask"""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Create output mask
        output = np.zeros_like(mask)

        # Keep only components larger than min_size
        for i in range(1, num_labels):  # Skip background (0)
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                output[labels == i] = 255

        return output

    def detect_lines_with_edges(self, binary_mask):
        """
        Detect line segments in binary mask and return edges too

        Parameters:
        - binary_mask: Binary image with white regions

        Returns:
        - lines: List of line segments with properties
        - edges: Edge detection result
        """
        # Detect edges with Canny
        edges = cv2.Canny(binary_mask, 30, 100)

        # Detect line segments using probabilistic Hough Transform
        line_segments = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=25,
            minLineLength=self.MIN_LINE_LENGTH,
            maxLineGap=self.MAX_LINE_GAP
        )

        lines = []
        if line_segments is not None:
            for segment in line_segments:
                x1, y1, x2, y2 = segment[0]

                # Calculate angle (0-180 degrees)
                dx = x2 - x1
                dy = y2 - y1

                if abs(dx) < 0.001:  # Vertical line
                    angle = 90.0
                else:
                    angle = math.degrees(math.atan2(dy, dx))
                    if angle < 0:
                        angle += 180

                # Calculate length
                length = math.sqrt(dx ** 2 + dy ** 2)

                # Calculate midpoint
                midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Calculate line endpoints (ordered for consistency)
                if x1 < x2 or (x1 == x2 and y1 < y2):
                    ordered_coords = (x1, y1, x2, y2)
                else:
                    ordered_coords = (x2, y2, x1, y1)

                lines.append({
                    'coords': ordered_coords,
                    'angle': angle,
                    'length': length,
                    'midpoint': midpoint,
                    'dx': dx,
                    'dy': dy
                })

        return lines, edges

    def save_all_lines_visualization(self, image, lines, filename):
        """
        Save visualization of all detected lines

        Parameters:
        - image: Original image
        - lines: List of detected lines
        - filename: Base filename
        """
        # Create visualization
        vis_image = image.copy()

        # Draw all lines
        for line in lines:
            x1, y1, x2, y2 = line['coords']
            # Color lines by angle
            angle = line['angle']
            if 0 <= angle < 30 or 150 <= angle <= 180:
                color = (0, 255, 0)  # Green for horizontal-ish
            elif 60 <= angle < 120:
                color = (0, 0, 255)  # Red for vertical-ish
            else:
                color = (255, 255, 0)  # Cyan for diagonal

            cv2.line(vis_image, (x1, y1), (x2, y2), color, 2)

            # Draw midpoint
            mx, my = line['midpoint']
            cv2.circle(vis_image, (mx, my), 3, (255, 255, 255), -1)

        # Add legend
        cv2.putText(vis_image, "Green: Horizontal-ish (0-30, 150-180 deg)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(vis_image, "Red: Vertical-ish (60-120 deg)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(vis_image, "Cyan: Diagonal (30-60, 120-150 deg)", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(vis_image, f"Total lines: {len(lines)}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imwrite(str(self.step_dirs['all_lines'] / f"{filename}_all_lines.png"), vis_image)

    def find_parallel_line_pairs(self, lines):
        """
        Find pairs of lines that are parallel to each other

        Parameters:
        - lines: List of line segments with properties

        Returns:
        - parallel_pairs: List of parallel line pairs
        """
        parallel_pairs = []

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1 = lines[i]
                line2 = lines[j]

                # Check if lines are parallel (within threshold)
                angle_diff = abs(line1['angle'] - line2['angle'])

                # Handle wrap-around at 180 degrees
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff

                if angle_diff < self.PARALLEL_THRESHOLD:
                    # Calculate perpendicular distance between lines
                    dist = self.calculate_perpendicular_distance(line1, line2)

                    # Check if distance is within double line range
                    if self.DISTANCE_THRESHOLD[0] <= dist <= self.DISTANCE_THRESHOLD[1]:
                        # Check if lines overlap
                        overlap_ratio = self.calculate_overlap_ratio(line1, line2)

                        if overlap_ratio >= self.MIN_OVERLAP_RATIO:
                            # Check if lines are in the same direction
                            if self.are_lines_same_direction(line1, line2):
                                parallel_pairs.append({
                                    'line1': line1,
                                    'line2': line2,
                                    'distance': dist,
                                    'angle': (line1['angle'] + line2['angle']) / 2,
                                    'overlap_ratio': overlap_ratio,
                                    'pair_score': self.calculate_pair_score(line1, line2, dist, overlap_ratio)
                                })

        return parallel_pairs

    def save_parallel_pairs_visualization(self, image, parallel_pairs, filename):
        """
        Save visualization of parallel line pairs

        Parameters:
        - image: Original image
        - parallel_pairs: List of parallel line pairs
        - filename: Base filename
        """
        vis_image = image.copy()

        # Draw parallel pairs
        for idx, pair in enumerate(parallel_pairs):
            # Use different colors for different pairs
            colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
            color = colors[idx % len(colors)]

            # Draw first line
            x1, y1, x2, y2 = pair['line1']['coords']
            cv2.line(vis_image, (x1, y1), (x2, y2), color, 2)

            # Draw second line
            x1, y1, x2, y2 = pair['line2']['coords']
            cv2.line(vis_image, (x1, y1), (x2, y2), color, 2)

            # Draw connecting line between midpoints
            mid1 = pair['line1']['midpoint']
            mid2 = pair['line2']['midpoint']
            cv2.line(vis_image, mid1, mid2, (255, 255, 255), 1)

            # Add label
            label = f"P{idx + 1}: d={pair['distance']:.1f}, o={pair['overlap_ratio']:.2f}"
            cv2.putText(vis_image, label, (mid1[0] + 5, mid1[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.putText(vis_image, f"Parallel pairs: {len(parallel_pairs)}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(str(self.step_dirs['parallel_pairs'] / f"{filename}_parallel_pairs.png"), vis_image)

    def calculate_perpendicular_distance(self, line1, line2):
        """
        Calculate perpendicular distance between two parallel lines

        Parameters:
        - line1, line2: Line dictionaries

        Returns:
        - distance: Perpendicular distance between lines
        """
        # Get line equations: ax + by + c = 0
        x1, y1, x2, y2 = line1['coords']
        x3, y3, x4, y4 = line2['coords']

        # Calculate line parameters for line1
        a1 = y2 - y1
        b1 = x1 - x2
        c1 = x2 * y1 - x1 * y2

        # Normalize
        norm1 = math.sqrt(a1 * a1 + b1 * b1)
        if norm1 < 0.001:
            return 0

        # Distance from midpoint of line2 to line1
        mx, my = line2['midpoint']
        distance = abs(a1 * mx + b1 * my + c1) / norm1

        return distance

    def calculate_overlap_ratio(self, line1, line2):
        """
        Calculate overlap ratio between two parallel lines

        Parameters:
        - line1, line2: Line dictionaries

        Returns:
        - overlap_ratio: Ratio of overlap (0-1)
        """
        x1_1, y1_1, x2_1, y2_1 = line1['coords']
        x1_2, y1_2, x2_2, y2_2 = line2['coords']

        # Project lines onto the direction perpendicular to their angle
        angle_rad = math.radians(line1['angle'])

        # Create rotation matrix
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        # Rotate points to align lines with x-axis
        def rotate_point(x, y):
            rx = x * cos_angle + y * sin_angle
            ry = -x * sin_angle + y * cos_angle
            return rx, ry

        # Rotate all points
        rx1_1, ry1_1 = rotate_point(x1_1, y1_1)
        rx2_1, ry2_1 = rotate_point(x2_1, y2_1)
        rx1_2, ry1_2 = rotate_point(x1_2, y1_2)
        rx2_2, ry2_2 = rotate_point(x2_2, y2_2)

        # Now lines are approximately horizontal, use x-coordinates for overlap
        min1, max1 = min(rx1_1, rx2_1), max(rx1_1, rx2_1)
        min2, max2 = min(rx1_2, rx2_2), max(rx1_2, rx2_2)

        # Calculate overlap
        overlap_start = max(min1, min2)
        overlap_end = min(max1, max2)

        if overlap_end > overlap_start:
            overlap_length = overlap_end - overlap_start
            line1_length = max1 - min1
            line2_length = max2 - min2

            # Return maximum overlap ratio
            return max(overlap_length / line1_length, overlap_length / line2_length)

        return 0.0

    def are_lines_same_direction(self, line1, line2):
        """
        Check if two lines have approximately the same direction

        Parameters:
        - line1, line2: Line dictionaries

        Returns:
        - True if lines point in the same direction
        """
        # Calculate dot product of direction vectors
        dx1, dy1 = line1['dx'], line1['dy']
        dx2, dy2 = line2['dx'], line2['dy']

        # Normalize vectors
        norm1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
        norm2 = math.sqrt(dx2 * dx2 + dy2 * dy2)

        if norm1 < 0.001 or norm2 < 0.001:
            return True

        # Calculate cosine similarity
        cos_similarity = (dx1 * dx2 + dy1 * dy2) / (norm1 * norm2)

        # Lines are considered same direction if cosine similarity > 0.7
        return cos_similarity > 0.7

    def calculate_pair_score(self, line1, line2, distance, overlap_ratio):
        """
        Calculate a quality score for a double line pair

        Parameters:
        - line1, line2: Line dictionaries
        - distance: Distance between lines
        - overlap_ratio: Overlap ratio

        Returns:
        - score: Higher is better
        """
        # Ideal distance is in the middle of the range
        ideal_distance = (self.DISTANCE_THRESHOLD[0] + self.DISTANCE_THRESHOLD[1]) / 2
        distance_score = 1.0 - abs(distance - ideal_distance) / ideal_distance

        # Length similarity score
        length_ratio = min(line1['length'], line2['length']) / max(line1['length'], line2['length'])

        # Angle similarity score
        angle_diff = abs(line1['angle'] - line2['angle'])
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        angle_score = 1.0 - angle_diff / self.PARALLEL_THRESHOLD

        # Combine scores
        total_score = (distance_score * 0.3 +
                       overlap_ratio * 0.3 +
                       length_ratio * 0.2 +
                       angle_score * 0.2)

        return total_score

    def filter_double_lines(self, parallel_pairs):
        """
        Filter parallel pairs to find true double solid lines

        Parameters:
        - parallel_pairs: List of parallel line pairs

        Returns:
        - double_lines: List of validated double line pairs
        """
        if not parallel_pairs:
            return []

        # Sort by pair score (highest first)
        sorted_pairs = sorted(parallel_pairs, key=lambda x: x['pair_score'], reverse=True)

        double_lines = []
        used_lines = set()

        for pair in sorted_pairs:
            line1_id = id(pair['line1'])
            line2_id = id(pair['line2'])

            # Check if lines have already been used
            if line1_id in used_lines or line2_id in used_lines:
                continue

            # Additional filtering
            line1_length = pair['line1']['length']
            line2_length = pair['line2']['length']

            # Both lines should be reasonably long
            if line1_length >= self.MIN_LINE_LENGTH and line2_length >= self.MIN_LINE_LENGTH:
                # Length ratio should be similar
                length_ratio = min(line1_length, line2_length) / max(line1_length, line2_length)
                if length_ratio > 0.6:  # Lines should be similar in length
                    double_lines.append(pair)
                    used_lines.add(line1_id)
                    used_lines.add(line2_id)

        return double_lines

    def save_double_lines_visualization(self, image, double_lines, filename):
        """
        Save visualization of double solid lines

        Parameters:
        - image: Original image
        - double_lines: List of double line pairs
        - filename: Base filename
        """
        vis_image = image.copy()

        # Draw double lines
        for idx, pair in enumerate(double_lines):
            # Use different colors for different double lines
            colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
                      (255, 255, 0), (0, 255, 127), (255, 0, 127), (127, 0, 255)]
            color = colors[idx % len(colors)]

            # Draw first line
            x1, y1, x2, y2 = pair['line1']['coords']
            cv2.line(vis_image, (x1, y1), (x2, y2), color, 3)

            # Draw second line
            x1, y1, x2, y2 = pair['line2']['coords']
            cv2.line(vis_image, (x1, y1), (x2, y2), color, 3)

            # Draw connecting line
            mid1 = pair['line1']['midpoint']
            mid2 = pair['line2']['midpoint']
            cv2.line(vis_image, mid1, mid2, (255, 255, 255), 1)

            # Add label
            label = f"DL{idx + 1}: d={pair['distance']:.1f}, s={pair['pair_score']:.2f}"
            cv2.putText(vis_image, label, (mid1[0] + 5, mid1[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.putText(vis_image, f"Double solid lines: {len(double_lines)}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imwrite(str(self.step_dirs['double_lines'] / f"{filename}_double_lines.png"), vis_image)

    def create_binary_output(self, image_shape, double_lines):
        """
        Create binary image with double solid lines on black background

        Parameters:
        - image_shape: Shape of original image
        - double_lines: List of double line pairs

        Returns:
        - binary_image: Black background with white double lines
        """
        # Create black background
        binary_image = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)

        # Draw double lines
        for pair in double_lines:
            # Draw first line
            x1, y1, x2, y2 = pair['line1']['coords']
            cv2.line(binary_image, (x1, y1), (x2, y2), 255, 3)

            # Draw second line
            x1, y1, x2, y2 = pair['line2']['coords']
            cv2.line(binary_image, (x1, y1), (x2, y2), 255, 3)

        return binary_image

    def create_comprehensive_visualization(self, image, intermediate_masks, edges,
                                           all_lines, parallel_pairs, double_lines,
                                           binary_image, filename):
        """
        Create a comprehensive visualization with all processing steps

        Parameters:
        - image: Original image
        - intermediate_masks: Dictionary of intermediate masks
        - edges: Edge detection result
        - all_lines: All detected lines
        - parallel_pairs: Parallel line pairs
        - double_lines: Double solid lines
        - binary_image: Final binary output
        - filename: Base filename
        """
        # Create a large composite image
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f'Double Solid Line Detection - {filename}', fontsize=16, fontweight='bold')

        # Plot 1: Original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('1. Original Image')
        axes[0, 0].axis('off')

        # Plot 2: HSV white mask
        axes[0, 1].imshow(intermediate_masks['hsv'], cmap='gray')
        axes[0, 1].set_title('2. HSV White Mask')
        axes[0, 1].axis('off')

        # Plot 3: Combined mask
        axes[0, 2].imshow(intermediate_masks['combined'], cmap='gray')
        axes[0, 2].set_title('3. Combined Mask')
        axes[0, 2].axis('off')

        # Plot 4: Cleaned mask
        axes[0, 3].imshow(intermediate_masks['cleaned'], cmap='gray')
        axes[0, 3].set_title('4. Cleaned Mask')
        axes[0, 3].axis('off')

        # Plot 5: Edges
        axes[1, 0].imshow(edges, cmap='gray')
        axes[1, 0].set_title('5. Edge Detection')
        axes[1, 0].axis('off')

        # Plot 6: All detected lines
        all_lines_vis = image.copy()
        for line in all_lines:
            x1, y1, x2, y2 = line['coords']
            cv2.line(all_lines_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        axes[1, 1].imshow(cv2.cvtColor(all_lines_vis, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'6. All Lines ({len(all_lines)})')
        axes[1, 1].axis('off')

        # Plot 7: Parallel pairs
        parallel_vis = image.copy()
        for idx, pair in enumerate(parallel_pairs[:8]):  # Show up to 8 pairs
            color = plt.cm.tab10(idx % 10)[:3]
            color = tuple(int(c * 255) for c in color)

            x1, y1, x2, y2 = pair['line1']['coords']
            cv2.line(parallel_vis, (x1, y1), (x2, y2), color, 2)

            x1, y1, x2, y2 = pair['line2']['coords']
            cv2.line(parallel_vis, (x1, y1), (x2, y2), color, 2)
        axes[1, 2].imshow(cv2.cvtColor(parallel_vis, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f'7. Parallel Pairs ({len(parallel_pairs)})')
        axes[1, 2].axis('off')

        # Plot 8: Double solid lines
        double_vis = image.copy()
        for idx, pair in enumerate(double_lines):
            color = plt.cm.Set1(idx % 9)[:3]
            color = tuple(int(c * 255) for c in color)

            x1, y1, x2, y2 = pair['line1']['coords']
            cv2.line(double_vis, (x1, y1), (x2, y2), color, 3)

            x1, y1, x2, y2 = pair['line2']['coords']
            cv2.line(double_vis, (x1, y1), (x2, y2), color, 3)
        axes[1, 3].imshow(cv2.cvtColor(double_vis, cv2.COLOR_BGR2RGB))
        axes[1, 3].set_title(f'8. Double Lines ({len(double_lines)})')
        axes[1, 3].axis('off')

        # Plot 9: Binary output
        axes[2, 0].imshow(binary_image, cmap='gray')
        axes[2, 0].set_title('9. Binary Output')
        axes[2, 0].axis('off')

        # Plot 10: Overlay on original
        overlay = image.copy()
        for idx, pair in enumerate(double_lines):
            color = plt.cm.Set1(idx % 9)[:3]
            color = tuple(int(c * 255) for c in color)

            x1, y1, x2, y2 = pair['line1']['coords']
            cv2.line(overlay, (x1, y1), (x2, y2), color, 3)

            x1, y1, x2, y2 = pair['line2']['coords']
            cv2.line(overlay, (x1, y1), (x2, y2), color, 3)
        axes[2, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2, 1].set_title('10. Overlay on Original')
        axes[2, 1].axis('off')

        # Plot 11: Statistics
        axes[2, 2].axis('off')
        stats_text = f"""Detection Statistics:
        • Total lines: {len(all_lines)}
        • Parallel pairs: {len(parallel_pairs)}
        • Double lines: {len(double_lines)}
        • Parameters:
          - Min line length: {self.MIN_LINE_LENGTH}px
          - Distance range: {self.DISTANCE_THRESHOLD[0]}-{self.DISTANCE_THRESHOLD[1]}px
          - Parallel threshold: {self.PARALLEL_THRESHOLD}°
          - Min overlap: {self.MIN_OVERLAP_RATIO}"""
        axes[2, 2].text(0.1, 0.5, stats_text, transform=axes[2, 2].transAxes,
                        fontsize=9, verticalalignment='center')

        # Plot 12: Legend
        axes[2, 3].axis('off')
        legend_text = """Color Legend:
        • Green: All detected lines
        • Various: Parallel pairs
        • Bright colors: Double solid lines

        Double Line Criteria:
        1. Parallel (within threshold)
        2. Close distance (5-25px)
        3. Overlap (>60%)
        4. Same direction
        5. Similar length"""
        axes[2, 3].text(0.1, 0.5, legend_text, transform=axes[2, 3].transAxes,
                        fontsize=9, verticalalignment='center')

        plt.tight_layout()
        plt.savefig(str(self.step_dirs['visualization'] / f"{filename}_comprehensive.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def pixel_to_geographic(self, pixel_coords, tile_info):
        """
        Convert pixel coordinates to geographic coordinates (WGS84)
        Using the SAME transformation as the downloader BBOX calculation

        Parameters:
        - pixel_coords: List of (x, y) pixel coordinates
        - tile_info: Dictionary with tile metadata and bbox

        Returns:
        - geo_coords: List of (lon, lat) coordinates
        """
        if tile_info is None or 'bbox' not in tile_info:
            return None

        # Get the Web Mercator bounding box from tile info
        minx, miny, maxx, maxy = tile_info['bbox']

        # Standard PNOA tile size
        img_width, img_height = 256, 256

        geo_coords = []
        for x_px, y_px in pixel_coords:
            # Convert pixel to Web Mercator (same as downloader BBOX calculation)
            x_wm = minx + (x_px / img_width) * (maxx - minx)
            y_wm = maxy - (y_px / img_height) * (maxy - miny)  # FLIP Y-AXIS

            # Convert Web Mercator to WGS84
            lon, lat = self.transformer.transform(x_wm, y_wm)
            geo_coords.append((lon, lat))

        return geo_coords

    def process_image(self, image_path):
        """
        Process a single image for double solid line detection

        Parameters:
        - image_path: Path to image file

        Returns:
        - binary_image: Binary image with detected double lines
        - double_lines: List of detected double line pairs
        - tile_info: Tile metadata
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error loading image: {image_path}")
            return None, [], None

        # Parse filename to get tile coordinates
        tile_info = self.parse_filename(image_path.name)

        if tile_info is None:
            print(f"Warning: Could not parse tile coordinates from {image_path.name}")
            print("Skipping coordinate conversion for this image")

        filename = Path(image_path).stem

        # Detect double solid lines with step-by-step images
        binary_image, double_lines = self.detect_double_solid_lines(image, filename)

        # Save binary image for compatibility
        binary_path = self.step_dirs['binary_images'] / f"double_lines_{filename}.png"
        cv2.imwrite(str(binary_path), binary_image)

        # Convert to geographic coordinates and add to GeoJSON
        if tile_info and double_lines:
            self.add_to_geojson(double_lines, tile_info, filename, str(image_path))

        return binary_image, double_lines, tile_info

    def add_to_geojson(self, double_lines, tile_info, filename, image_path):
        """
        Add double lines to GeoJSON features

        Parameters:
        - double_lines: List of detected double line pairs
        - tile_info: Tile metadata
        - filename: Image filename
        - image_path: Full image path
        """
        for idx, pair in enumerate(double_lines):
            # Get coordinates for both lines
            line1_coords = pair['line1']['coords']
            line2_coords = pair['line2']['coords']

            # Convert to geographic coordinates
            line1_pixels = [(line1_coords[0], line1_coords[1]),
                            (line1_coords[2], line1_coords[3])]
            line2_pixels = [(line2_coords[0], line2_coords[1]),
                            (line2_coords[2], line2_coords[3])]

            line1_geo = self.pixel_to_geographic(line1_pixels, tile_info)
            line2_geo = self.pixel_to_geographic(line2_pixels, tile_info)

            if line1_geo and line2_geo:
                # Feature for first line
                feature1 = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [line1_geo[0][0], line1_geo[0][1]],
                            [line1_geo[1][0], line1_geo[1][1]]
                        ]
                    },
                    "properties": {
                        "image": filename,
                        "tile_x": tile_info['x'],
                        "tile_y": tile_info['y'],
                        "tile_z": tile_info['z'],
                        "double_line_id": f"{filename}_{idx}",
                        "line_position": "first",
                        "length_pixels": float(pair['line1']['length']),
                        "angle_degrees": float(pair['angle']),
                        "distance_between": float(pair['distance']),
                        "overlap_ratio": float(pair['overlap_ratio']),
                        "pair_score": float(pair['pair_score']),
                        "source_image": image_path
                    }
                }

                # Feature for second line
                feature2 = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [line2_geo[0][0], line2_geo[0][1]],
                            [line2_geo[1][0], line2_geo[1][1]]
                        ]
                    },
                    "properties": {
                        "image": filename,
                        "tile_x": tile_info['x'],
                        "tile_y": tile_info['y'],
                        "tile_z": tile_info['z'],
                        "double_line_id": f"{filename}_{idx}",
                        "line_position": "second",
                        "length_pixels": float(pair['line2']['length']),
                        "angle_degrees": float(pair['angle']),
                        "distance_between": float(pair['distance']),
                        "overlap_ratio": float(pair['overlap_ratio']),
                        "pair_score": float(pair['pair_score']),
                        "source_image": image_path
                    }
                }

                self.all_double_lines.append(feature1)
                self.all_double_lines.append(feature2)

    def process_folder(self, input_folder, extensions=('.png',)):
        """
        Process all images in a folder

        Parameters:
        - input_folder: Path to folder containing images
        - extensions: Image file extensions to process

        Returns:
        - results: Dictionary of processing results
        """
        input_path = Path(input_folder)
        if not input_path.exists():
            print(f"Error: Folder '{input_folder}' does not exist.")
            return {}

        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))

        print(f"Found {len(image_files)} images in {input_folder}")
        print(f"Expected filename format: pnoa_z_x_y.png")
        print(f"Step-by-step images will be saved in: {self.output_dir}")
        print(f"{'=' * 60}")

        results = {}
        total_double_lines = 0

        # Process each image
        for img_path in tqdm(image_files, desc="Detecting double solid lines"):
            print(f"\nProcessing: {img_path.name}")

            # Parse filename
            tile_info = self.parse_filename(img_path.name)

            if tile_info:
                print(f"  Tile: z={tile_info['z']}, x={tile_info['x']}, y={tile_info['y']}")

            # Process image
            binary_image, double_lines, _ = self.process_image(img_path)

            if binary_image is not None:
                num_pairs = len(double_lines)
                total_double_lines += num_pairs

                results[img_path.name] = {
                    'success': True,
                    'binary_path': str(self.step_dirs['binary_images'] / f"double_lines_{img_path.stem}.png"),
                    'num_double_pairs': num_pairs,
                    'num_total_lines': num_pairs * 2,
                    'tile_info': tile_info
                }

                if num_pairs > 0:
                    print(f"  ✓ Found {num_pairs} double solid line pairs")
                    print(f"  ✓ Step-by-step images saved to {self.output_dir}/[01-09]_*/")
                else:
                    print(f"  ✗ No double lines found")
                    print(f"  ✓ Step-by-step images still saved for debugging")
            else:
                results[img_path.name] = {'success': False}
                print(f"  ✗ Failed to process")

        # Generate GeoJSON
        geojson_path = self.generate_geojson()

        # Generate summary
        self.generate_summary(results, total_double_lines)

        return results, geojson_path

    def generate_geojson(self):
        """Generate GeoJSON file from all detected double lines"""
        if not self.all_double_lines:
            print("\nNo double lines detected to create GeoJSON")
            return None

        # Create GeoJSON FeatureCollection
        geojson = {
            "type": "FeatureCollection",
            "name": "PNOA_Double_Solid_Lines",
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:OGC:1.3:CRS84"  # WGS84
                }
            },
            "features": self.all_double_lines
        }

        # Save GeoJSON
        geojson_path = self.output_dir / "pnoa_double_solid_lines.geojson"
        with open(geojson_path, 'w') as f:
            json.dump(geojson, f, indent=2)

        print(f"\n✓ GeoJSON saved to: {geojson_path}")
        print(f"  Total line features: {len(self.all_double_lines)}")
        print(f"  Total double line pairs: {len(self.all_double_lines) // 2}")

        return geojson_path

    def generate_summary(self, results, total_double_pairs):
        """Generate summary report"""
        successful = sum(1 for r in results.values() if r.get('success', False))
        total_lines = total_double_pairs * 2

        print(f"\n{'=' * 60}")
        print("DOUBLE SOLID LINE DETECTION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total images processed: {len(results)}")
        print(f"Successfully processed: {successful}")
        print(f"Total double line pairs found: {total_double_pairs}")
        print(f"Total individual lines: {total_lines}")

        if successful > 0:
            avg_pairs = total_double_pairs / successful
            print(f"Average pairs per image: {avg_pairs:.1f}")

        print(f"\nDetection parameters:")
        print(f"  Min line length: {self.MIN_LINE_LENGTH}px")
        print(f"  Parallel threshold: {self.PARALLEL_THRESHOLD}°")
        print(f"  Distance range: {self.DISTANCE_THRESHOLD[0]}-{self.DISTANCE_THRESHOLD[1]}px")
        print(f"  Min overlap ratio: {self.MIN_OVERLAP_RATIO}")

        print(f"\nOutput directory structure:")
        for step_name, step_dir in self.step_dirs.items():
            if step_dir.exists():
                num_files = len(list(step_dir.glob("*.png")))
                step_num = step_name.split('_')[0] if step_name[0].isdigit() else ""
                step_desc = step_name.split('_', 1)[1] if '_' in step_name else step_name
                print(f"  {step_num:2} {step_dir.name}: {num_files:3d} images")

        print(f"\nGeoJSON file: {self.output_dir}/pnoa_double_solid_lines.geojson")
        print(f"{'=' * 60}")

        # Save summary to file
        summary_file = self.output_dir / "detection_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("PNOA DOUBLE SOLID LINE DETECTION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Date: {np.datetime64('now')}\n\n")

            f.write(f"Detection parameters:\n")
            f.write(f"  Min line length: {self.MIN_LINE_LENGTH}px\n")
            f.write(f"  Parallel threshold: {self.PARALLEL_THRESHOLD}°\n")
            f.write(f"  Distance range: {self.DISTANCE_THRESHOLD[0]}-{self.DISTANCE_THRESHOLD[1]}px\n")
            f.write(f"  Min overlap ratio: {self.MIN_OVERLAP_RATIO}\n\n")

            f.write(f"Results:\n")
            f.write(f"  Total images: {len(results)}\n")
            f.write(f"  Successfully processed: {successful}\n")
            f.write(f"  Total double line pairs: {total_double_pairs}\n")
            f.write(f"  Total individual lines: {total_lines}\n\n")

            f.write("Image-wise results:\n")
            f.write("-" * 40 + "\n")
            for img_name, result in results.items():
                if result['success']:
                    f.write(f"{img_name}: {result['num_double_pairs']} pairs")
                    if 'tile_info' in result and result['tile_info']:
                        ti = result['tile_info']
                        f.write(f" (z={ti['z']}, x={ti['x']}, y={ti['y']})")
                    f.write("\n")
                else:
                    f.write(f"{img_name}: FAILED\n")


# Command line usage
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect double solid lines in PNOA imagery and generate GeoJSON with step-by-step images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/pnoa/images
  %(prog)s /path/to/pnoa/images --min-length 50 --distance-min 10

Expected filename format: pnoa_z_x_y.png
Example: pnoa_18_123456_789012.png

Output includes step-by-step images for debugging.
        """
    )

    parser.add_argument("input_folder", help="Folder containing PNOA images (pnoa_z_x_y.png format)")
    parser.add_argument("--output", default="double_solid_lines_output", help="Output directory")
    parser.add_argument("--min-length", type=int, default=40, help="Minimum line length (pixels)")
    parser.add_argument("--distance-min", type=int, default=1, help="Minimum distance between lines (pixels)")
    parser.add_argument("--distance-max", type=int, default=5, help="Maximum distance between lines (pixels)")
    parser.add_argument("--parallel-threshold", type=float, default=5,
                        help="Max angle difference for parallel lines (degrees)")
    parser.add_argument("--overlap-ratio", type=float, default=0.6, help="Minimum overlap ratio (0-1)")

    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        sys.exit(1)

    # Create detector with custom parameters
    detector = DoubleSolidLineDetector(args.output)
    detector.MIN_LINE_LENGTH = args.min_length
    detector.DISTANCE_THRESHOLD = (args.distance_min, args.distance_max)
    detector.PARALLEL_THRESHOLD = args.parallel_threshold
    detector.MIN_OVERLAP_RATIO = args.overlap_ratio

    print(f"PNOA Double Solid Line Detector with Step-by-Step Images")
    print(f"Filename format: pnoa_z_x_y.png")
    print(f"Input folder: {args.input_folder}")
    print(f"Output directory: {args.output}")
    print(f"Detection parameters:")
    print(f"  Min line length: {detector.MIN_LINE_LENGTH}px")
    print(f"  Line distance: {detector.DISTANCE_THRESHOLD[0]}-{detector.DISTANCE_THRESHOLD[1]}px")
    print(f"  Parallel threshold: {detector.PARALLEL_THRESHOLD}°")
    print(f"  Min overlap ratio: {detector.MIN_OVERLAP_RATIO}")
    print(f"{'=' * 60}")

    # Process folder
    results, geojson_path = detector.process_folder(args.input_folder)