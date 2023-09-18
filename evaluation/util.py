from collections import defaultdict
from data.regions_japan import *
from datasets.dataset import split_location_token


def get_countries_occurring_in(locations: list) -> list:
    countries = []
    for location in locations:
        country, site = split_location_token(location)
        countries.append(country)
    return list(set(countries))


def group_items_per_location(locations: list,
                             items: list,
                             ) -> defaultdict:
    assert len(locations) == len(items)

    groups = defaultdict(list)

    for location, item in zip(locations, items):
        groups[location].append(item)

    return groups


def group_items_per_country(locations: list,
                            items: list,
                            ) -> defaultdict:
    assert len(locations) == len(items)

    groups = defaultdict(list)

    for location, item in zip(locations, items):
        country, site = split_location_token(location)
        groups[country].append(item)

    return groups


def group_items_per_region_japan(locations: list,
                                 items: list,
                                 ) -> defaultdict:
    assert len(locations) == len(items)

    groups = defaultdict(list)

    for location, item in zip(locations, items):
        country, site = split_location_token(location)
        if country != 'Japan':
            continue
        region_id = LOCATIONS[location]
        groups[region_id].append(item)

    return groups
