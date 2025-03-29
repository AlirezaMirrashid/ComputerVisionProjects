import requests
import math
import os
from PIL import Image
import io
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box, Polygon, LineString, mapping, MultiLineString

import osmnx as ox

def latlon_to_tile(lat, lon, zoom):
    """Convert latitude/longitude to tile (x, y) at a given zoom level."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return int(x), int(y)


def download_tile(x, y, zoom, save_dir):
    """Download a single tile and return it as an image."""
    # MAP
    # url = f'https://mt0.google.com/vt?x={x}&y={y}&z={zoom}'
    # SAT
    url = f'https://mt0.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}'

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        # image_path = os.path.join(save_dir, f'tile_{x}_{y}.png')
        # image.save(image_path)
        return image
    else:
        print(f"Failed to download tile {x}, {y}, status code: {response.status_code}")
        return None


def download_images_for_bounding_box(lat_min, lon_min, lat_max, lon_max, zoom, save_dir):
    """Download Google satellite images for a bounding box at a given zoom level."""
    x_min, y_min = latlon_to_tile(lat_max, lon_min, zoom)  # southwest corner
    x_max, y_max = latlon_to_tile(lat_min, lon_max, zoom)  # northeast corner



    tiles = []
    for x in range(x_min, x_max + 1):
        for y in range(y_max, y_min - 1, -1):  # y decreases as we move north
            tile_image = download_tile(x, y, zoom, save_dir)
            if tile_image:
                tiles.append((tile_image, x, y))

    return tiles, x_min, y_max, x_max, y_min


def stitch_images(tiles, x_min, y_max, x_max, y_min):
    """Stitch downloaded tiles into a single image."""
    tile_width, tile_height = tiles[0][0].size
    width = (x_max - x_min + 1) * tile_width
    height = (y_max - y_min + 1) * tile_height

    stitched_image = Image.new('RGB', (width, height))
    for tile_image, x, y in tiles:
        x_pos = (x - x_min) * tile_width
        y_pos = (y - y_min) * tile_height
        stitched_image.paste(tile_image, (x_pos, y_pos))

    return stitched_image


def save_stitched_image_with_georeference(stitched_image, x_min, y_max, x_max, y_min, zoom, output_path):
    """Save the stitched image as a GeoTIFF with georeferencing."""
    # Calculate geotransform
    tile_width, tile_height = stitched_image.size
    lat_min, lon_min = tile_to_latlon(x_min, y_min, zoom)  # Top-left corner of bounding box
    lat_max, lon_max = tile_to_latlon(x_max + 1, y_max + 1, zoom)  # Bottom-right corner of bounding box

    # Pixel size in degrees
    pixel_width = (lon_max - lon_min) / tile_width
    pixel_height = (lat_max - lat_min) / tile_height

    # Set up geotransform (origin X, pixel width, rotation, origin Y, rotation, pixel height)
    geotransform = (
        lon_min,  # top-left x (longitude)
        pixel_width,  # pixel width
        0,  # rotation (0 if north-up)
        lat_max,  # top-left y (latitude)
        0,  # rotation (0 if north-up)
        pixel_height  # pixel height (negative because y decreases as latitude increases)
    )

    # Convert PIL image to numpy array
    image_array = np.array(stitched_image)

    # Set up the GeoTIFF file
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, tile_width, tile_height, 3, gdal.GDT_Byte, options=['COMPRESS=LZW'])

    # Set the geotransform and projection
    dataset.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)  # EPSG:3857 for Web Mercator
    dataset.SetProjection(srs.ExportToWkt())

    # Write each color band
    dataset.GetRasterBand(1).WriteArray(image_array[:, :, 0])  # Red channel
    dataset.GetRasterBand(2).WriteArray(image_array[:, :, 1])  # Green channel
    dataset.GetRasterBand(3).WriteArray(image_array[:, :, 2])  # Blue channel

    # Save and close the file
    dataset.FlushCache()
    dataset = None
    print(f"GeoTIFF saved as {output_path}")


def tile_to_latlon(x, y, zoom):
    """Convert tile x, y at a given zoom level to latitude/longitude."""
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

count = 0
class Downloader:
    def __init__(self):
        self.count = 0
    def main(self, lat_min, lon_min, lat_max, lon_max, zoom, save_dir):
        tiles, x_min, y_max, x_max, y_min = download_images_for_bounding_box(lat_min, lon_min, lat_max, lon_max, zoom, save_dir)
        stitched_image = stitch_images(tiles, x_min, y_max, x_max, y_min)

        # Define tags to capture buildings, pools, playgrounds, and similar structures
        tags = {
            'building': True,
            # 'leisure': ['swimming_pool', 'stadium', 'playground', 'track','sports_centre','pitch'],
            # 'amenity': ['university'],
            # 'amenity': ['parking', 'bicycle_parking', 'motorcycle_parking'],

            # 'landuse': ['retail', 'commercial', 'industrial', 'residential']
        }



        try:
            gdf = ox.geometries_from_bbox(lat_max, lat_min, lon_max, lon_min, tags)

            # Convert to 2D if there are any 3D geometries (e.g., dropping elevation data)
            gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom if geom.is_empty else geom.simplify(0))

            img_width, img_height = stitched_image.size

            # Define affine transform for the raster based on the bounding box
            transform = rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, img_width, img_height)

            # Rasterize the geometries to create a binary mask
            mask = rasterize(
                [(geom, 1) for geom in gdf.geometry],  # Use value 1 for features
                out_shape=(img_height, img_width),
                transform=transform,
                fill=0,  # Fill background with 0
                dtype=np.uint8
            )
            Image.fromarray((mask*255).astype("uint8")).save(os.path.join(save_dir,f"mask_buildings_{self.count}.png"))
            stitched_image.save(os.path.join(save_dir, f"image_{self.count}.png"))
            self.count += 1
        except:
            pass
        # save_stitched_image_with_georeference(stitched_image, x_min, y_max, x_max, y_min, zoom, output_path)


if __name__ == '__main__':
    """
    Caltech
    -118.13399036029298
    34.13234718959901 
    -118.12122617983148
    34.1426965596546

    LACC
    -118.27465054038804 
    34.03630057028926 
    -118.26494224952235 
    34.046026133279646

    LAHG
    -118.11996769039479 
    34.12342598553937 
    -118.10844966728995 
    34.133037154215266

    Rose Bowl Stadium    
    -118.17423308307235 34.14684310903734 -118.16226539128057 34.174962824324446

    SMC
    -118.47612807961325 34.01412138535203 -118.46728807031162 34.01927598094681

    """
    # min_lon, min_lat, max_lon, max_lat
    lat_min = 34.1
    lat_max = 34.2
    lon_min = -118.2
    lon_max = -118.1
    zoom = 19
    save_dir = "./dataset_osm"
    os.makedirs(save_dir,exist_ok=True)
    step = 0.001
    downloader = Downloader()
    for i in range(100):
        for j in range(100):

            downloader.main(lat_min+i*step, lon_min+j*step, lat_min+(i+1)*step, lon_min+(j+1)*step, zoom, save_dir)
