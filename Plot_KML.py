import matplotlib.pyplot as plt
from pathlib import Path
from fastkml import kml
from shapely.geometry import LineString
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# Load the KML file
def Plot_KML(kml_file_path):
    input_filename = os.path.basename(kml_file_path)
    id = input_filename.split(".")[0]
    id = f"{id} Survey Line"

    with open(kml_file_path, 'rb') as f:
        k = kml.KML()
        k.from_string(f.read())

    # Extract the coordinates of the placemarks
    lats, lons = [], []
    for feature in k.features():
        for placemark in feature.features():
            if isinstance(placemark.geometry, LineString):
                for coord in placemark.geometry.coords:
                    lats.append(coord[1])
                    lons.append(coord[0])

    # Determine the extent of the plot
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    extent = [lon_min-0.5, lon_max+0.5, lat_min-0.5, lat_max+0.5]

    # Plot the coordinates using Matplotlib
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.plot(lons, lats, transform=ccrs.PlateCarree(), color="red", label=id)
    #ax.coastlines(resolution="10m")

    land_110m = cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='face', facecolor=cfeature.COLORS['land'])
    ax.add_feature(land_110m)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.top_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    gl.zorder=0
    # Add a title and labels to the axes
    ax.set_title(id)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    # Add a title to the plot

    plt.show() 
    return print("> Done")


