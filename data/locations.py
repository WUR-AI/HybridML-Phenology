# import os
#
# import pandas as pd
#
#
# from .merra_v2 import DATA_PATH_MERRA
#
#
# DATA_FILENAME = 'locations.csv'
#
#
# # TODO -- include altitudes
#
# def get_data(set_index=True):
#     """
#     Get a pandas DataFrame containing all locations with corresponding coordinates, as well as their index in the
#     MERRA-v2 data grid.
#     The DataFrame has the following columns:
#      - location: location name (str)
#      - lat: latitude (float)
#      - lon: longitude (float)
#      - lat_geos5_native: latitude converted to index in the MERRA-v2 dataset
#      - lon_geos5_native: longitude converted to index in the MERRA-v2 dataset
#
#     :return: a pd.DataFrame containing location data
#     """
#     path = os.path.join(DATA_PATH_MERRA, DATA_FILENAME)
#     df = pd.read_csv(path, index_col=0,)
#     if set_index:
#         df.set_index('location', inplace=True)
#     return df
#
#
# def get_location_tokens() -> list:
#     df = get_data(set_index=True)
#     return list(df.index)
#
#
# def split_location_token(location: str):
#     """
#     Split a location token string into (country, site)
#     :param location: the location token
#     :return: a two-tuple of (location (str), site (str))
#     """
#     country, site = location.split('/')
#     return country, site
#
#
# def get_location_tokens_of_country(country: str) -> list:
#     tokens = [t for t in get_location_tokens() if split_location_token(t)[0] == country]
#     return tokens
#
#
# def get_location_tokens_japan():
#     return get_location_tokens_of_country('Japan')
#
#
# def get_location_tokens_south_korea():
#     return get_location_tokens_of_country('South Korea')
#
#
# def get_location_tokens_switzerland():
#     return get_location_tokens_of_country('Switzerland')
#
#
# def get_location_tokens_usa():
#     return get_location_tokens_of_country('USA')
#
#
# if __name__ == '__main__':
#
#     _data = get_data()
#
#     print(_data)
#     print(_data.loc['Japan/Naze']['lat'])
#
