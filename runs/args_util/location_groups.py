
import data.bloom_doy
import data.regions_japan

"""
    Pre-configured named location groups on which the models can be run
    Defines a mapping from
        group_name -> list of location tokens in group
"""

LOCATION_GROUPS = {
    'all': data.bloom_doy.get_locations(),
    'japan': data.bloom_doy.get_locations_japan(),
    'switzerland': data.bloom_doy.get_locations_switzerland(),
    'south_korea': data.bloom_doy.get_locations_south_korea(),
    'usa': data.bloom_doy.get_locations_usa(),
    'japan_south_korea': data.bloom_doy.get_locations_japan() + data.bloom_doy.get_locations_south_korea(),
    'japan_switzerland': data.bloom_doy.get_locations_japan() + data.bloom_doy.get_locations_switzerland(),
    'no_us': data.bloom_doy.get_locations_japan() + data.bloom_doy.get_locations_switzerland() + data.bloom_doy.get_locations_south_korea(),

    'japan_hokkaido': list(data.regions_japan.LOCATIONS_HOKKAIDO.keys()),
    'japan_tohoku': list(data.regions_japan.LOCATIONS_TOHOKU.keys()),
    'japan_hokuriku': list(data.regions_japan.LOCATIONS_HOKURIKU.keys()),
    'japan_kanto_koshin': list(data.regions_japan.LOCATIONS_KANTO_KOSHIN.keys()),
    'japan_kinki': list(data.regions_japan.LOCATIONS_KINKI.keys()),
    'japan_chugoku': list(data.regions_japan.LOCATIONS_CHUGOKU.keys()),
    'japan_tokai': list(data.regions_japan.LOCATIONS_TOKAI.keys()),
    'japan_shikoku': list(data.regions_japan.LOCATIONS_SHIKOKU.keys()),
    'japan_kyushu_north': list(data.regions_japan.LOCATIONS_KYUSHU_NORTH.keys()),
    'japan_kyushu_south_amami': list(data.regions_japan.LOCATIONS_KYUSHU_SOUTH_AMAMI.keys()),
    'japan_okinawa': list(data.regions_japan.LOCATIONS_OKINAWA.keys()),

    'japan_wo_okinawa': list(data.regions_japan.LOCATIONS_WO_OKINAWA.keys()),

    'japan_known_variety': list(data.regions_japan.LOCATION_VARIETY_JAPAN.keys()),  # Locations in Japan for which the variety is known
    'japan_yedoenis': list(data.regions_japan.LOCATIONS_JAPAN_YEDOENIS.keys()),
    'japan_sargentii': list(data.regions_japan.LOCATIONS_JAPAN_SARGENTII.keys()),

    'japan_yedoenis_sargentii': list(data.regions_japan.LOCATIONS_JAPAN_YEDOENIS.keys()) + list(data.regions_japan.LOCATIONS_JAPAN_SARGENTII.keys()),

    'japan_yedoenis_south_korea': list(data.regions_japan.LOCATIONS_JAPAN_YEDOENIS.keys()) + list(data.regions_japan.LOCATION_VARIETY_SOUTH_KOREA.keys()),

}
