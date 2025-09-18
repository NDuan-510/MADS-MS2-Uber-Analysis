"""
This module handling data collection step for the project. It includes functions 
that work with API, and cut big files into small file for storage.
"""


import requests, time, logging, re
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import datacommons as dc
import datacommons_pandas as dd

def generate_address(
        location_list,
        api_key,
        country_code='IN',
        return_size=5
        ):
    
    json_data = []
    success_loc = []
    fail_loc = []

    for loc in tqdm(location_list):
        r = requests.get(
            'https://api.openrouteservice.org/geocode/search?'\
            f'api_key={api_key}&text={loc}&boundary.country={country_code}&size={return_size}'
            )
        try:
            n = len(r.json()['features']) 
            if n >= 1:  # return at least 1 result
                json_data.extend(r.json()['features'])
                success_loc += [loc]*n
            else:
                fail_loc.append(loc)
        except:
            fail_loc.append(loc)
        time.sleep(0.5)  # too much API call will break the connection
    print('List of locations failed to generate address:')
    print(fail_loc)

    loc_meta_df = pd.json_normalize(json_data)[['properties.source_id','geometry.coordinates',
                                                'properties.label','properties.region','properties.county',
                                                'properties.locality']]
    loc_meta_df[['longitude','latitude']] = loc_meta_df['geometry.coordinates'].apply(pd.Series)
    loc_meta_df['uber_loc'] = success_loc

    loc_meta_df.rename(columns = {
        'properties.source_id':'source_id',
        'properties.label':'address',
        'properties.region':'region',
        'properties.county':'county',
        'properties.locality':'locality'
    },inplace=True)
    loc_meta_df = loc_meta_df[['source_id','longitude','latitude','uber_loc',
                               'address','region','county','locality']]

    return loc_meta_df

def find_best_address(record,loc_data=None):
    pickup = record['Pickup Location']
    dropoff = record['Drop Location']
    ride_distance = record['Ride Distance']
    avail_locs = loc_data['uber_loc'].unique()
    if (pickup not in avail_locs) or (dropoff not in avail_locs):
        return np.full(10,np.nan)
    
    start_df = loc_data.loc[loc_data['uber_loc']==pickup]
    end_df = loc_data.loc[loc_data['uber_loc']==dropoff]
    start_coords = np.radians(start_df[['longitude','latitude']])
    end_coords = np.radians(end_df[['longitude','latitude']])

    # match pair and find distance between 2 locations
    # read this paper if necessary https://towardsdatascience.com/using-scikit-learns-binary-trees-to-efficiently-find-latitude-and-longitude-neighbors-909979bd929b/
    tree = BallTree(end_coords, metric='haversine')
    distances, indices = tree.query(start_coords, k=1)
    if np.isnan(ride_distance):
        start_idx = np.argmin(distances)
    else:
        start_idx = np.argmin(np.abs(distances - ride_distance))
    end_idx = indices[start_idx][0]

    start_info = start_df[['longitude','latitude',
                          'address','region','locality']].iloc[start_idx].to_numpy()
    end_info = end_df[['longitude','latitude',
                          'address','region','locality']].iloc[end_idx].to_numpy()
    
    return np.concatenate((start_info,end_info))


def generate_weather_data(location_df):
    weather_df = pd.DataFrame()

    for lat,long,address in tqdm(location_df.to_numpy()):
        r = requests.get(f'https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={long}&start_date=2024-01-01&end_date=2024-12-31&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation,rain,snowfall,wind_speed_10m')
        try:
            combine_keys = list(r.json()['hourly'].keys())

            data_df = pd.DataFrame()
            for key in combine_keys:
                add_df = pd.json_normalize(r.json(),record_path=['hourly',key])
                data_df = pd.concat([data_df,add_df],axis=1)
            data_df.columns = combine_keys
            data_df.insert(1,'address',address)

            weather_df = pd.concat([weather_df,data_df],axis=0)
            weather_df['time'] = pd.to_datetime(weather_df['time'])
        except:
            print(f'Fail to retrieve data for: {address}')
            continue

        time.sleep(2)
    
    return weather_df

def collect_DataCommons_city_id(api_key,country_DCID = "country/IND", progress_bar=True):
    """
    Collect city DCID code from DataCommon API, which will be used to get feature data of each city.

    Notes
    ----------
    This function is modified. But credit is given to team 01-mboarts-minhkha-yryuzaki-el-2024fall from Milestone 1 project.

    Parameters
    ----------
    api_key: str 
        DataCommons API key use to request data.

    country_DCID: str
        Choose which country to get city information from.

    progress_bar: bool 
        Turn on/ off the progress bar for the data retrieving step.

    Returns
    ----------
    city_df: pd.DataFrame
        City names and its DCID code.    
    """

    dc.set_api_key(api_key)

    # Generate DCID and state names
    states = dc.get_places_in([country_DCID], "State")[country_DCID]
    states = dd.get_property_values(states, "name")
    state_df = pd.DataFrame(
        [(item[0], item[1][0]) for item in states.items()],
        columns=["state_dcid", "state_name"],
    )

    city_df = pd.DataFrame()

    if progress_bar:
        iterator = tqdm(state_df.to_numpy(), desc="States", position=0)
    else:
        iterator = state_df.to_numpy()

    # Generate DCID and name for cities
    for state_dcid, state_name in tqdm(state_df.to_numpy(), desc="States", position=0):
        # Use DataCommons package to request data
        try:
            city_ids = dc.get_places_in([state_dcid], "City")[state_dcid]
            city_names = dc.get_property_values(city_ids, "name")

            city_names = [
                (key, values[0]) for key, values in city_names.items() if values != []
            ]
            buffer_df = pd.DataFrame(city_names, columns=["city_dcid", "city_name"])
            buffer_df.insert(0, "state_name", state_name)
            buffer_df.insert(0, "state_dcid", state_dcid)
            city_df = pd.concat((city_df, buffer_df))
        except:
            print(f'Fail for {state_dcid}-{state_name}')

    city_df.reset_index(drop=True, inplace=True)

    return city_df

def create_logger(name="root", filename_path="default.log", logging_level=logging.INFO):
    """
    Create a log to hold terminal output if necessary. This is used for data collection step to understand the error, if any.

    Notes
    ----------
    This function is modified. But credit is given to team 01-mboarts-minhkha-yryuzaki-el-2024fall from Milestone 1 project.

    Parameters
    ----------
    name: str 
        Name of the request person, use for record.

    filename_path: str
        Path to store the log file.

    logging_level: logging object
        Set log level.

    Returns
        logging object use to handle inputs.
    -------
    """
    logger = logging.getLogger(name)

    # Clear any handlers of previously created logging instance so it won't target the same logging file.
    logger.handlers.clear()

    logger.setLevel(logging_level)

    # File to store log outputs
    fhandler = logging.FileHandler(filename=filename_path)
    # Log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.propagate = False

    return logger

def collect_DataCommons_features_data(
    api_key,
    state_dcid_list,
    dc_features,
    feature_name_map,
    batch_size=5,
    logger_name="root",
    log_file="logging/datacommon.log",
    progress_bar=True,
):
    """
    Collect features data from DataCommon API.

    Notes
    ----------
    This function is modified. But credit is given to team 01-mboarts-minhkha-yryuzaki-el-2024fall from Milestone 1 project.

    Parameters
    ----------
    api_key: str
        DataCommons API key use to request data.

    state_dcid_list: list
        List of unique dcid code to retrieve feature information.

    feature_name_map: dict
        Mapping to transform feature names to a more readable format.

    batch_size: int
        Number of city to request for data at a time.

    logger_name: str
        The logger name in when printing to log file.

    log_file: str
        Path to there to store the log file.

    progress_bar: bool
        Turn on/ off the progress bar for the data retrieving step.

    Returns
    ----------
    data_df: pd.DataFrame
    """
    dc.set_api_key(api_key)

    n_city = len(state_dcid_list)

    # Number of cut sections, big query to dd.build_time_series_dataframe cause DataCommons API to exclude lots of data, so
    # we reduce the number of cities data request by separate them into batches.
    n_sections = n_city // batch_size

    idx_list = np.arange(n_city)
    slice_sections = np.array_split(idx_list, n_sections)

    city_id_array = np.array(state_dcid_list)

    # Create logging handler
    logger = create_logger(logger_name, log_file)

    data_df = pd.DataFrame()

    if progress_bar:
        outer_iterator = tqdm(slice_sections, desc="Batch", position=0)
    else:
        outer_iterator = slice_sections

    i = 0
    dc_features = list(feature_name_map.keys())

    for batch in outer_iterator:
        process_city_list = city_id_array[batch]

        # Use to hold intermediate df
        buffer_df = pd.DataFrame({"unique_conn_id": []})

        if progress_bar:
            inner_iterator = tqdm(dc_features, desc="Feature", position=1, leave=False)
        else:
            inner_iterator = slice_sections

        for feature in inner_iterator:
            try:
                feature_df = dd.build_time_series_dataframe(process_city_list, feature)
                if feature_df.shape[0] == 0:
                    continue
                # The datacommon package is quite old, it return two version of data, so there are duplicate rows, the first to
                # appear seems to be always better, so we group by 'place' and use head(1) to get the first record.
                feature_df = feature_df.groupby("place").head(1)

                feature_df.reset_index(inplace=True)
                # melt the dataframe to long format
                feature_df = feature_df.melt(
                    "place", var_name="year", value_name=feature_name_map[feature]
                )

                # Insert unique_conn_id column to join dataframe
                feature_df.insert(
                    0, "unique_conn_id", feature_df["place"] + "-" + feature_df["year"]
                )
                feature_df.drop(["place", "year"], axis=1, inplace=True)

                buffer_df = pd.merge(
                    buffer_df, feature_df, how="outer", on=["unique_conn_id"]
                )

            except Exception as error:  # We may have error because properly one of the cites in the batch didn't have data.
                
                # Try to request the feature for each individual city in the batch.
                for city_id in process_city_list:
                    try:
                        feature_sery = dd.build_time_series(city_id, feature)
                        if feature_sery.size == 0:
                            continue
                        feature_df = pd.DataFrame(
                            {
                                "year": feature_sery.index,
                                feature_name_map[feature]: feature_sery.values,
                            }
                        )
                        feature_df.insert(0, "place", city_id)

                        # Insert unique_conn_id column to join dataframe
                        feature_df.insert(
                            0,
                            "unique_conn_id",
                            feature_df["place"] + "-" + feature_df["year"],
                        )
                        feature_df.drop(["place", "year"], axis=1, inplace=True)

                        buffer_df = pd.merge(
                            buffer_df, feature_df, how="outer", on=["unique_conn_id"]
                        )

                        # Try to slow down quick requests, there maybe a request/s limit for DataCommon APIs.
                        time.sleep(0.5)
                    except:
                        # Capture which city didn't have data
                        logger.info(
                            f"Error when retrieving data for Batch No.{i} - city: {city_id} - feature: {feature}"
                        )
                        logger.error(error)

        if buffer_df.shape[0] > 0:  # If batch df have data, join with the main df
            data_df = pd.concat((data_df, buffer_df))
        i += 1

    data_df.rename(columns={"place": "state_dcid"}, inplace=True)

    # Retrieve year & state_dcid from unique_conn_id column
    data_df.insert(0, "year", data_df["unique_conn_id"].str.extract(r"-(.*)"))
    data_df.insert(0, "state_dcid", data_df["unique_conn_id"].str.extract(r"(.*)-"))

    # Some records use YYYY-MM-DD format instead of YYYY, which cause unwanted string in the state_dcid and year columns
    def year_format_fix(string):
        r = re.search("([0-9]{4})-[0-9]{2}-[0-9]{2}", string)
        if r:
            # Get YYYY group only
            return r.groups(0)[0]
        else:
            return string

    def fix_state_dcid(string):
        r = re.match(("^([a-zA-Z//0-9]+)-"), string)
        if r:
            # Get group related to state_dcid only
            return r.groups(0)[0]
        else:
            return string

    # Fix year column
    data_df["year"] = data_df["year"].astype("str").apply(lambda x: year_format_fix(x))
    data_df["year"] = data_df["year"].astype(int)

    # Fix state_dcid column
    data_df["state_dcid"] = data_df["state_dcid"].apply(lambda x: fix_state_dcid(x))

    data_df.drop(["unique_conn_id"], axis=1, inplace=True)

    data_df.sort_values(["state_dcid", "year"], inplace=True)
    data_df.reset_index(drop=True, inplace=True)

    feature_cols = data_df.columns.tolist()
    # Something there different versions of same column when merge dataframes due to inconsistence in API response using DataCommon package
    # e.g.: population,population_x,population_y, we try to combine those feature columns by calculate it means.
    for column in feature_cols:
        # Check if any column have '_x' at the end
        if re.match(r".*[_x]$", column):
            replace_col = column[:-2]
            try:
                # Calculation mean
                data_df[replace_col] = data_df[
                    [replace_col, replace_col + "_x", replace_col + "_y"]
                ].mean(numeric_only=True, skipna=True, axis=1)
                # Drop those extra columns
                data_df.drop(
                    [replace_col + "_x", replace_col + "_y"], axis=1, inplace=True
                )

            except Exception as error:
                logger.info(f"Error when combining data for '{replace_col}' column.")
                logger.error(error)

    return data_df