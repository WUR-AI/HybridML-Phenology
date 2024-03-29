
import data.bloom_doy
import data.regions


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

    'japan_hokkaido': list(data.regions.LOCATIONS_HOKKAIDO.keys()),
    'japan_tohoku': list(data.regions.LOCATIONS_TOHOKU.keys()),
    'japan_hokuriku': list(data.regions.LOCATIONS_HOKURIKU.keys()),
    'japan_kanto_koshin': list(data.regions.LOCATIONS_KANTO_KOSHIN.keys()),
    'japan_kinki': list(data.regions.LOCATIONS_KINKI.keys()),
    'japan_chugoku': list(data.regions.LOCATIONS_CHUGOKU.keys()),
    'japan_tokai': list(data.regions.LOCATIONS_TOKAI.keys()),
    'japan_shikoku': list(data.regions.LOCATIONS_SHIKOKU.keys()),
    'japan_kyushu_north': list(data.regions.LOCATIONS_KYUSHU_NORTH.keys()),
    'japan_kyushu_south_amami': list(data.regions.LOCATIONS_KYUSHU_SOUTH_AMAMI.keys()),
    'japan_okinawa': list(data.regions.LOCATIONS_OKINAWA.keys()),

    'japan_wo_okinawa': list(data.regions.LOCATIONS_WO_OKINAWA.keys()),

    'japan_known_variety': list(data.regions.LOCATION_VARIETY_JAPAN.keys()),  # Locations in Japan for which the variety is known
    'japan_yedoensis': list(data.regions.LOCATIONS_JAPAN_YEDOENSIS.keys()),
    'japan_sargentii': list(data.regions.LOCATIONS_JAPAN_SARGENTII.keys()),

    'japan_yedoensis_sargentii': list(data.regions.LOCATIONS_JAPAN_YEDOENSIS.keys()) + list(data.regions.LOCATIONS_JAPAN_SARGENTII.keys()),

    'japan_yedoenis_south_korea': list(data.regions.LOCATIONS_JAPAN_YEDOENSIS.keys()) + list(data.regions.LOCATION_VARIETY_SOUTH_KOREA.keys()),

    'japan_tokyo': ['Japan/Tokyo'],
    'japan_kyoto': ['Japan/Kyoto-1'],

    'south_korea_wo_juji': data.regions.LOCATIONS_SOUTH_KOREA_WO_JUJI,

    # All selected locations to be included in comparing the process-based models with the learned chill model
    # Includes:
    #   - Japan (only locations with yedoensis and sargentii cultivars)
    #   - Switzerland
    #   - South Korea
    'selection': list(data.regions.LOCATIONS_JAPAN_YEDOENSIS.keys()) + list(data.regions.LOCATIONS_JAPAN_SARGENTII.keys()) + data.bloom_doy.get_locations_switzerland() + data.bloom_doy.get_locations_south_korea(),

}
