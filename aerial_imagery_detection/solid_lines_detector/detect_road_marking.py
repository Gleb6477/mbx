import os
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import warnings
import math

warnings.filterwarnings('ignore')
from pyproj import Transformer


class StraightEdgeDetector:
    def __init__(self, output_dir="straight_edge_detection_output"):
        """
        Detects only straight edges by filtering out curvy/tree edges

        Parameters:
        - output_dir: Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize coordinate transformer
        self.transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

        # For tracking all straight edges across all images
        self.all_edges_geojson = []

        # Canny edge detection parameters
        self.CANNY_LOW_THRESHOLD = 50
        self.CANNY_HIGH_THRESHOLD = 150

        # Filtering parameters for straightness
        self.MIN_LINEARITY = 0.7  # 0-1, higher = more straight (1 = perfectly straight line)
        self.MIN_CONTOUR_LENGTH = 30  # Minimum contour length in pixels
        self.MAX_ASPECT_RATIO = 10  # Maximum width/height ratio for lines

        # Hough line parameters (for verification)
        self.HOUGH_THRESHOLD = 20
        self.MIN_LINE_LENGTH = 30
        self.MAX_LINE_GAP = 10

        # Create subdirectories
        self.create_directories()

    def create_directories(self):
        """Create output directories"""
        self.dirs = {
            'original': self.output_dir / "01_original",
            'edges': self.output_dir / "02_edges",
            'contours': self.output_dir / "03_contours",
            'straight_edges': self.output_dir / "04_straight_edges",
            'hough_lines': self.output_dir / "05_hough_lines",
            'final': self.output_dir / "06_final"
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)

    def detect_straight_edges(self, image, filename):
        """
        Detect and filter only straight edges (not tree edges)

        Parameters:
        - image: Input image (BGR format)
        - filename: Base filename for saving

        Returns:
        - straight_edges: Binary image with only straight edges
        - hough_lines: List of detected Hough lines for verification
        """
        # Save original image
        cv2.imwrite(str(self.dirs['original'] / f"{filename}_original.png"), image)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

        # Step 1: Apply Canny edge detection
        edges = cv2.Canny(blurred, self.CANNY_LOW_THRESHOLD, self.CANNY_HIGH_THRESHOLD)
        cv2.imwrite(str(self.dirs['edges'] / f"{filename}_edges.png"), edges)

        # Step 2: Find contours in the edge image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create visualization of all contours
        contour_vis = np.zeros_like(edges)
        cv2.drawContours(contour_vis, contours, -1, 255, 1)
        cv2.imwrite(str(self.dirs['contours'] / f"{filename}_contours.png"), contour_vis)

        # Step 3: Filter contours to keep only straight ones
        straight_edges = self.filter_straight_contours(contours, edges.shape)

        # Save filtered straight edges
        cv2.imwrite(str(self.dirs['straight_edges'] / f"{filename}_straight_edges.png"), straight_edges)

        # Step 4: Apply Hough transform on straight edges (for verification)
        hough_lines = self.detect_hough_lines(straight_edges)

        # Create Hough line visualization
        hough_vis = image.copy()
        for line in hough_lines:
            x1, y1, x2, y2 = line['coords']
            cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(hough_vis, f"Hough lines: {len(hough_lines)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(self.dirs['hough_lines'] / f"{filename}_hough_lines.png"), hough_vis)

        # Step 5: Create final output - FIXED VERSION
        # Create black background with same shape as input image
        final_image = np.zeros_like(image)

        # Create a mask for straight edges
        mask = straight_edges > 0

        # Apply white color to all channels where mask is True
        final_image[mask] = (255, 255, 255)

        cv2.imwrite(str(self.dirs['final'] / f"{filename}_final.png"), final_image)

        return straight_edges, hough_lines

    def filter_straight_contours(self, contours, image_shape):
        """
        Filter contours to keep only straight edges

        Parameters:
        - contours: List of contours from findContours
        - image_shape: Shape of the image

        Returns:
        - straight_mask: Binary mask with only straight edges
        """
        # Create empty mask
        straight_mask = np.zeros(image_shape, dtype=np.uint8)

        for contour in contours:
            # Skip very small contours
            if len(contour) < self.MIN_CONTOUR_LENGTH:
                continue

            # Method 1: Check linearity using approximation
            # Approximate the contour with a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # A straight line should be approximated by 2 points
            if len(approx) <= 4:  # Allow some tolerance (2-4 points)
                # Method 2: Check bounding rectangle aspect ratio
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]

                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)

                    # Method 3: Check if contour is roughly a line
                    # Fit a line to the contour points
                    if len(contour) >= 2:
                        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

                        # Calculate linearity: how well points fit to a line
                        # For a perfect line, all points would be on the fitted line
                        linearity = self.calculate_linearity(contour, vx[0], vy[0], x[0], y[0])

                        if linearity > self.MIN_LINEARITY and aspect_ratio > 2:
                            # Draw this contour as a straight edge
                            cv2.drawContours(straight_mask, [contour], -1, 255, 1)

            # Alternative simpler method: Check if contour can be fitted with a line
            elif len(contour) >= 10:  # Need enough points for meaningful line fitting
                # Fit line using RANSAC-like approach
                if self.is_contour_straight(contour):
                    cv2.drawContours(straight_mask, [contour], -1, 255, 1)

        return straight_mask

    def calculate_linearity(self, contour, vx, vy, x0, y0):
        """
        Calculate how well contour points fit to a line

        Parameters:
        - contour: Contour points
        - vx, vy: Line direction vector
        - x0, y0: Point on the line

        Returns:
        - linearity: 0-1, higher = more linear
        """
        if len(contour) < 3:
            return 0.0

        # Calculate distances of each point to the fitted line
        distances = []
        for point in contour:
            x, y = point[0]
            # Distance from point (x,y) to line defined by (x0,y0) and direction (vx,vy)
            numerator = abs((y0 - y) * vx - (x0 - x) * vy)
            denominator = np.sqrt(vx ** 2 + vy ** 2)
            if denominator > 0:
                distance = numerator / denominator
                distances.append(distance)

        if not distances:
            return 0.0

        # Calculate average distance and normalize
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)

        if max_distance > 0:
            # Lower average distance = more linear
            linearity = 1.0 - (avg_distance / max_distance)
            return max(0.0, min(1.0, linearity))

        return 1.0  # All points are exactly on the line

    def is_contour_straight(self, contour, angle_tolerance=15):
        """
        Simple check if contour is approximately straight

        Parameters:
        - contour: Contour points
        - angle_tolerance: Maximum angle variation in degrees

        Returns:
        - is_straight: Boolean indicating if contour is straight
        """
        if len(contour) < 10:
            return False

        # Get contour points
        points = contour.reshape(-1, 2)

        # Calculate overall direction using first and last points
        start_point = points[0]
        end_point = points[-1]

        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]

        if abs(dx) < 1 and abs(dy) < 1:
            return False

        # Calculate overall angle
        overall_angle = math.degrees(math.atan2(dy, dx))
        if overall_angle < 0:
            overall_angle += 180

        # Check if consecutive segments have similar angles
        angle_changes = []
        for i in range(len(points) - 1):
            dx_seg = points[i + 1][0] - points[i][0]
            dy_seg = points[i + 1][1] - points[i][1]

            if abs(dx_seg) < 0.1 and abs(dy_seg) < 0.1:
                continue

            seg_angle = math.degrees(math.atan2(dy_seg, dx_seg))
            if seg_angle < 0:
                seg_angle += 180

            # Calculate angle difference
            angle_diff = abs(seg_angle - overall_angle)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff

            angle_changes.append(angle_diff)

        if not angle_changes:
            return False

        # Check if most segments have similar angles
        avg_angle_change = np.mean(angle_changes)
        return avg_angle_change < angle_tolerance

    def detect_hough_lines(self, edge_image):
        """
        Apply Hough line transform to detect straight lines

        Parameters:
        - edge_image: Binary image with edges

        Returns:
        - lines: List of detected lines with properties
        """
        line_segments = cv2.HoughLinesP(
            edge_image,
            rho=1,
            theta=np.pi / 180,
            threshold=self.HOUGH_THRESHOLD,
            minLineLength=self.MIN_LINE_LENGTH,
            maxLineGap=self.MAX_LINE_GAP
        )

        lines = []
        if line_segments is not None:
            for segment in line_segments:
                x1, y1, x2, y2 = segment[0]

                # Calculate angle
                dx = x2 - x1
                dy = y2 - y1

                if abs(dx) < 0.001:
                    angle = 90.0
                else:
                    angle = math.degrees(math.atan2(dy, dx))
                    if angle < 0:
                        angle += 180

                # Calculate length
                length = math.sqrt(dx ** 2 + dy ** 2)

                lines.append({
                    'coords': (x1, y1, x2, y2),
                    'angle': angle,
                    'length': length
                })

        return lines

    def parse_filename(self, filename):
        """
        Parse pnoa_z_x_y.png filename format
        """
        try:
            stem = Path(filename).stem
            parts = stem.split('_')

            if len(parts) != 4:
                return None

            z = int(parts[1])
            x = int(parts[2])
            y = int(parts[3])

            # Calculate bounding box
            bbox = self.calculate_bbox_from_tile(x, y, z)

            return {
                'z': z,
                'x': x,
                'y': y,
                'bbox': bbox,
                'filename': filename
            }

        except (ValueError, IndexError):
            return None

    def calculate_bbox_from_tile(self, x, y, z):
        """Calculate Web Mercator bounding box from tile coordinates"""
        n = 2.0 ** z
        lon_west = x / n * 360.0 - 180.0
        lon_east = (x + 1) / n * 360.0 - 180.0

        lat_north_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat_north = math.degrees(lat_north_rad)

        lat_south_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
        lat_south = math.degrees(lat_south_rad)

        earth_radius = 6378137.0

        minx = lon_west * (earth_radius * math.pi / 180.0)
        maxx = lon_east * (earth_radius * math.pi / 180.0)

        miny = math.log(math.tan((90 + lat_south) * math.pi / 360.0)) * earth_radius
        maxy = math.log(math.tan((90 + lat_north) * math.pi / 360.0)) * earth_radius

        return [minx, miny, maxx, maxy]

    def pixel_to_geographic(self, pixel_coords, tile_info):
        """Convert pixel coordinates to geographic coordinates"""
        if tile_info is None or 'bbox' not in tile_info:
            return None

        minx, miny, maxx, maxy = tile_info['bbox']
        img_width, img_height = 256, 256

        geo_coords = []
        for x_px, y_px in pixel_coords:
            x_wm = minx + (x_px / img_width) * (maxx - minx)
            y_wm = maxy - (y_px / img_height) * (maxy - miny)

            lon, lat = self.transformer.transform(x_wm, y_wm)
            geo_coords.append((lon, lat))

        return geo_coords

    def process_image(self, image_path):
        """Process a single image for straight edge detection"""
        image = cv2.imread(str(image_path))
        if image is None:
            return None, [], None

        tile_info = self.parse_filename(image_path.name)
        filename = Path(image_path).stem

        # Detect straight edges
        straight_edges, hough_lines = self.detect_straight_edges(image, filename)

        # For GeoJSON, we'll use Hough lines (they're already straight)
        if tile_info and hough_lines:
            self.add_to_geojson(hough_lines, tile_info, filename, str(image_path))

        return straight_edges, hough_lines, tile_info

    def add_to_geojson(self, lines, tile_info, filename, image_path):
        """Add lines to GeoJSON features"""
        for idx, line in enumerate(lines):
            x1, y1, x2, y2 = line['coords']

            pixels = [(x1, y1), (x2, y2)]
            geo_coords = self.pixel_to_geographic(pixels, tile_info)

            if geo_coords:
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [geo_coords[0][0], geo_coords[0][1]],
                            [geo_coords[1][0], geo_coords[1][1]]
                        ]
                    },
                    "properties": {
                        "image": filename,
                        "tile_x": tile_info['x'],
                        "tile_y": tile_info['y'],
                        "tile_z": tile_info['z'],
                        "line_id": f"{filename}_{idx}",
                        "length_pixels": float(line['length']),
                        "angle_degrees": float(line['angle']),
                        "source_image": image_path
                    }
                }

                self.all_edges_geojson.append(feature)

    def process_folder(self, input_folder, extensions=('.png',)):
        """Process all images in a folder"""
        input_path = Path(input_folder)
        if not input_path.exists():
            print(f"Error: Folder '{input_folder}' does not exist.")
            return {}

        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))

        print(f"Found {len(image_files)} images")
        print(f"Filtering out curvy/tree edges, keeping only straight edges")
        print(f"{'=' * 60}")

        results = {}
        total_edges = 0
        total_lines = 0

        for img_path in tqdm(image_files, desc="Filtering straight edges"):
            print(f"\nProcessing: {img_path.name}")

            _, lines, tile_info = self.process_image(img_path)

            if lines:
                total_lines += len(lines)
                results[img_path.name] = {
                    'success': True,
                    'num_lines': len(lines),
                    'tile_info': tile_info
                }
                print(f"  Found {len(lines)} straight lines")
            else:
                results[img_path.name] = {'success': False}
                print(f"  No straight lines found")

        # Generate GeoJSON
        geojson_path = self.generate_geojson()

        return results, geojson_path

    def generate_geojson(self):
        """Generate GeoJSON file from straight edges"""
        if not self.all_edges_geojson:
            print("\nNo straight edges detected to create GeoJSON")
            return None

        geojson = {
            "type": "FeatureCollection",
            "name": "PNOA_Straight_Edges",
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
                }
            },
            "features": self.all_edges_geojson
        }

        geojson_path = self.output_dir / "straight_edges.geojson"
        with open(geojson_path, 'w') as f:
            json.dump(geojson, f, indent=2)

        print(f"\n✓ GeoJSON saved to: {geojson_path}")
        print(f"  Total straight edge features: {len(self.all_edges_geojson)}")

        return geojson_path


# Alternative simpler version for creating final image
def create_final_output_simple(straight_edges, image_shape):
    """
    Alternative simpler way to create final output image

    Parameters:
    - straight_edges: Binary image with straight edges
    - image_shape: Shape of original image

    Returns:
    - final_image: Black background with white edges (3-channel BGR)
    """
    # Method 1: Convert grayscale to BGR
    final_image = cv2.cvtColor(straight_edges, cv2.COLOR_GRAY2BGR)

    # Method 2: Create black image and draw white edges
    # final_image = np.zeros((*image_shape[:2], 3), dtype=np.uint8)
    # final_image[straight_edges > 0] = (255, 255, 255)

    return final_image


# Command line usage with argparse
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect only straight edges by filtering out curvy/tree edges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python straight_edge_detector.py /path/to/pnoa/images
  python straight_edge_detector.py /path/to/pnoa/images --output ./my_output
  python straight_edge_detector.py /path/to/pnoa/images --canny-low 30 --canny-high 100

Expected filename format: pnoa_z_x_y.png
        """
    )

    parser.add_argument("--input_folder", help="Folder containing PNOA images (pnoa_z_x_y.png format)")
    parser.add_argument("--output", default="straight_edge_detection_output", help="Output directory")
    parser.add_argument("--canny-low", type=int, default=40, help="Canny low threshold")
    parser.add_argument("--canny-high", type=int, default=120, help="Canny high threshold")
    parser.add_argument("--min-linearity", type=float, default=0.2, help="Minimum linearity (0-1)")
    parser.add_argument("--min-length", type=int, default=15, help="Minimum contour length (pixels)")

    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        exit(1)

    # Create detector with custom parameters
    detector = StraightEdgeDetector(args.output)
    detector.CANNY_LOW_THRESHOLD = args.canny_low
    detector.CANNY_HIGH_THRESHOLD = args.canny_high
    detector.MIN_LINEARITY = args.min_linearity
    detector.MIN_CONTOUR_LENGTH = args.min_length

    print(f"STRAIGHT EDGE DETECTOR - Filter out curvy/tree edges")
    print(f"Input folder: {args.input_folder}")
    print(f"Output directory: {args.output}")
    print(f"Detection parameters:")
    print(f"  Canny thresholds: {detector.CANNY_LOW_THRESHOLD}-{detector.CANNY_HIGH_THRESHOLD}")
    print(f"  Min linearity: {detector.MIN_LINEARITY}")
    print(f"  Min contour length: {detector.MIN_CONTOUR_LENGTH}px")
    print(f"{'=' * 60}")

    # Process folder
    results, geojson_path = detector.process_folder(args.input_folder)

    if geojson_path:
        print(f"\n✓ Processing complete!")
        print(f"\nFiles created in {args.output}:")
        print(f"  01_original/      - Original images")
        print(f"  02_edges/         - All Canny edges")
        print(f"  03_contours/      - Detected contours")
        print(f"  04_straight_edges/- Filtered straight edges only")
        print(f"  05_hough_lines/   - Hough lines from straight edges")
        print(f"  06_final/         - Final output (white edges on black)")
        print(f"  straight_edges.geojson - GeoJSON file")