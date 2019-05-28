# -*- coding: utf-8 -*-
"""
Module with helper functions to expand on some features of geopandas.
"""

import os

import fiona
import geopandas as gpd

def read_file(filepath: str,
              layer: str = 'default',
              columns: [] = None) -> gpd.GeoDataFrame:
    """
    Reads a file to a pandas dataframe. The fileformat is detected based on the filepath extension.

    # TODO: think about if possible/how to support  adding optional parameter and pass them to next function, example encoding, float_format,...
    """
    _, ext = os.path.splitext(filepath)

    ext_lower = ext.lower()
    if ext_lower == '.shp':
        return gpd.read_file(filepath)
    elif ext_lower == '.gpkg':
        return gpd.read_file(filepath, layer=layer)
    else:
        raise Exception(f"Not implemented for extension {ext_lower}")

def to_file(gdf: gpd.GeoDataFrame,
            filepath: str,
            layer: str = 'default',
            index: bool = True):
    """
    Reads a pandas dataframe to file. The fileformat is detected based on the filepath extension.

    # TODO: think about if possible/how to support  adding optional parameter and pass them to next function, example encoding, float_format,...
    """
    _, ext = os.path.splitext(filepath)

    ext_lower = ext.lower()
    if ext_lower == '.shp':
        gdf.to_file(filepath)
    elif ext_lower == '.gpkg':
        gdf.to_file(filepath, layer=layer, driver="GPKG")
    else:
        raise Exception(f"Not implemented for extension {ext_lower}")
        
def get_crs(filepath):
    with fiona.open(filepath, 'r') as geofile:
        return geofile.crs

def is_geofile(filepath) -> bool:
    """
    Determines based on the filepath if this is a geofile.
    """
    _, file_ext = os.path.splitext(filepath)
    return is_geofile_ext(file_ext)

def is_geofile_ext(file_ext) -> bool:
    """
    Determines based on the file extension if this is a geofile.
    """
    file_ext_lower = file_ext.lower()
    if file_ext_lower in ('.shp', '.gpkg', '.geojson'):
        return True
    else:
        return False
