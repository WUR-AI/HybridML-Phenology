
import os

import pandas as pd

from config import PATH_DATA_DIR

"""

    Functions for loading the original competition data

"""

# Source directory for all original competition data
DATA_PATH_ORIGINAL = os.path.join(PATH_DATA_DIR, 'original')

# Individual file names
FILENAME_JAPAN = 'japan.csv'
FILENAME_KYOTO = 'kyoto.csv'
FILENAME_LIESTAL = 'liestal.csv'
FILENAME_METEOSWISS = 'meteoswiss.csv'
FILENAME_SOUTH_KOREA = 'south_korea.csv'
FILENAME_WASHINGTONDC = 'washingtondc.csv'

FILENAME_USA_NPN_PHENOMETRICS = 'USA-NPN_individual_phenometrics_data.csv'
FILENAME_USA_NPN_PHENOMETRICS_DESC = 'USA-NPN_individual_phenometrics_datafield_descriptions.csv'
FILENAME_USA_NPN_STATUS_INTENSITY = 'USA-NPN_status_intensity_observations_data.csv'
FILENAME_USA_NPN_STATUS_INTENSITY_DESC = 'USA-NPN_status_intensity_datafield_descriptions.csv'

# Paths to data files
DATA_PATH_JAPAN = f'{DATA_PATH_ORIGINAL}/{FILENAME_JAPAN}'
DATA_PATH_KYOTO = f'{DATA_PATH_ORIGINAL}/{FILENAME_KYOTO}'
DATA_PATH_LIESTAL = f'{DATA_PATH_ORIGINAL}/{FILENAME_LIESTAL}'
DATA_PATH_METEOSWISS = f'{DATA_PATH_ORIGINAL}/{FILENAME_METEOSWISS}'
DATA_PATH_SOUTH_KOREA = f'{DATA_PATH_ORIGINAL}/{FILENAME_SOUTH_KOREA}'
DATA_PATH_WASHINGTONDC = f'{DATA_PATH_ORIGINAL}/{FILENAME_WASHINGTONDC}'

DATA_PATH_USA_NPN_PHENOMETRICS = f'{DATA_PATH_ORIGINAL}/{FILENAME_USA_NPN_PHENOMETRICS}'
DATA_PATH_USA_NPN_PHENOMETRICS_DESC = f'{DATA_PATH_ORIGINAL}/{FILENAME_USA_NPN_PHENOMETRICS_DESC}'
DATA_PATH_USA_NPN_STATUS_INTENSITY = f'{DATA_PATH_ORIGINAL}/{FILENAME_USA_NPN_STATUS_INTENSITY}'
DATA_PATH_USA_NPN_STATUS_INTENSITY_DESC = f'{DATA_PATH_ORIGINAL}/{FILENAME_USA_NPN_STATUS_INTENSITY_DESC}'

COLUMNS = ('location', 'lat', 'long', 'alt', 'year', 'bloom_date', 'bloom_doy')


def get_data_japan() -> pd.DataFrame:
    """

    ['location', 'lat', 'long', 'alt', 'year', 'bloom_date', 'bloom_doy']

    :return:
    """
    df = pd.read_csv(DATA_PATH_JAPAN)
    return df


def get_data_kyoto() -> pd.DataFrame:
    """

    ['location', 'lat', 'long', 'alt', 'year', 'bloom_date', 'bloom_doy']

    :return:
    """
    df = pd.read_csv(DATA_PATH_KYOTO)
    return df


def get_data_liestal() -> pd.DataFrame:
    """

    ['location', 'lat', 'long', 'alt', 'year', 'bloom_date', 'bloom_doy']

    :return:
    """
    df = pd.read_csv(DATA_PATH_LIESTAL)
    return df


def get_data_meteoswiss() -> pd.DataFrame:
    """

    ['location', 'lat', 'long', 'alt', 'year', 'bloom_date', 'bloom_doy']

    :return:
    """
    df = pd.read_csv(DATA_PATH_METEOSWISS)
    return df


def get_data_south_korea() -> pd.DataFrame:
    """

    ['location', 'lat', 'long', 'alt', 'year', 'bloom_date', 'bloom_doy']

    :return:
    """
    df = pd.read_csv(DATA_PATH_SOUTH_KOREA)
    return df


def get_data_washingtondc() -> pd.DataFrame:
    """

    ['location', 'lat', 'long', 'alt', 'year', 'bloom_date', 'bloom_doy']

    :return:
    """
    df = pd.read_csv(DATA_PATH_WASHINGTONDC)
    return df


def get_individual_phenometrics_data() -> pd.DataFrame:
    """

    ['Site_ID', 'Latitude', 'Longitude', 'Elevation_in_Meters', 'State',
       'Species_ID', 'Genus', 'Species', 'Common_Name', 'Kingdom',
       'Individual_ID', 'Phenophase_ID', 'Phenophase_Description',
       'First_Yes_Year', 'First_Yes_Month', 'First_Yes_Day', 'First_Yes_DOY',
       'First_Yes_Julian_Date', 'NumDays_Since_Prior_No', 'Last_Yes_Year',
       'Last_Yes_Month', 'Last_Yes_Day', 'Last_Yes_DOY',
       'Last_Yes_Julian_Date', 'NumDays_Until_Next_No', 'AGDD', 'AGDD_in_F',
       'Tmax_Winter', 'Tmax_Spring', 'Tmin_Winter', 'Tmin_Spring',
       'Prcp_Winter', 'Prcp_Spring']

    :return:
    """
    df = pd.read_csv(DATA_PATH_USA_NPN_PHENOMETRICS)
    return df


def get_individual_phenometrics_datafield_descriptions() -> pd.DataFrame:
    """

    ['Field name', 'Field description', 'Controlled value choices']

    :return:
    """
    df = pd.read_csv(DATA_PATH_USA_NPN_PHENOMETRICS_DESC)
    return df


def get_status_intensity_observations_data() -> pd.DataFrame:
    """

    ['Observation_ID', 'Update_Datetime', 'Site_ID', 'Latitude', 'Longitude',
       'Elevation_in_Meters', 'State', 'Species_ID', 'Genus', 'Species',
       'Common_Name', 'Kingdom', 'Individual_ID', 'Phenophase_ID',
       'Phenophase_Description', 'Observation_Date', 'Day_of_Year',
       'Phenophase_Status', 'Intensity_Category_ID', 'Intensity_Value',
       'Abundance_Value', 'AGDD', 'AGDD_in_F', 'Tmax_Winter', 'Tmax_Spring',
       'Tmin_Winter', 'Tmin_Spring', 'Prcp_Winter', 'Prcp_Spring',
       'Daylength']

    :return:
    """
    df = pd.read_csv(DATA_PATH_USA_NPN_STATUS_INTENSITY)
    return df


def get_status_intensity_datafield_descriptions() -> pd.DataFrame:
    """

    ['Field name', 'Field description', 'Controlled value choices']

    :return:
    """
    df = pd.read_csv(DATA_PATH_USA_NPN_STATUS_INTENSITY_DESC)
    return df


if __name__ == '__main__':

    print(__file__)
    print(os.path.abspath(__file__))
    print(os.path.abspath(os.path.join(__file__, os.pardir)))
