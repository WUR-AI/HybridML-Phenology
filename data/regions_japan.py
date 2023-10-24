
"""

    Climate regions are defined by the Japan Meteorological Agency
    https://www.data.jma.go.jp/stats/data/en/index.html

"""
from data.bloom_doy import get_locations_south_korea

REGIONS_JAPAN = {
    0: 'Hokkaido',
    1: 'Tohoku',
    2: 'Hokuriku',
    3: 'Kanto-Koshin',
    4: 'Kinki',
    5: 'Chugoku',
    6: 'Tokai',
    7: 'Shikoku',
    8: 'Kyushu-North',
    9: 'Kyushu-South-Amami',
    10: 'Okinawa',
}

"""
    Map locations to climate region ids
"""

LOCATIONS_REGIONS_JAPAN = {
    'Japan/Kushiro-1': 0,
    'Japan/Naze': 10,  # Officially classified as 9 -- climate is closer to 10
    'Japan/Miyakejima': 3,
    'Japan/Fukushima': 1,
    'Japan/Kagoshima': 9,
    'Japan/Saga': 8,
    'Japan/Nagasaki': 8,
    'Japan/Muroran-2': 0,
    'Japan/Tottori-2': 5,
    'Japan/Niigata': 2,
    'Japan/Toyooka': 4,
    'Japan/Nagoya-1': 6,
    'Japan/Tateyama': 2,
    'Japan/Tottori-1': 5,
    'Japan/Nago': 10,
    'Japan/Kobe': 4,
    'Japan/Tsuruga': 2,
    'Japan/Shimonoseki': 8,
    'Japan/Miyako': 1,
    'Japan/Izuhara-2': 8,
    'Japan/Shionomisaki': 4,
    'Japan/Asahikawa': 0,
    'Japan/Wakkanai': 0,
    'Japan/Fukue': 8,
    'Japan/Sendai-2': 1,
    'Japan/Mito': 3,
    'Japan/Nemuro': 0,
    'Japan/Wajima': 2,
    'Japan/Rumoi': 0,
    'Japan/Gifu': 3,
    'Japan/Hachinohe': 1,
    'Japan/Yamagata': 1,
    'Japan/Abashiri': 0,
    'Japan/Kyoto-2': 4,
    'Japan/Onahama': 1,
    'Japan/Matsuyama': 7,
    'Japan/Muroran-1': 0,
    'Japan/Oita': 8,
    'Japan/Ishigakijima': 10,
    'Japan/Esashi': 0,
    'Japan/Kushiro-2': 0,
    'Japan/Kochi-1': 7,
    'Japan/Okayama': 5,
    'Japan/Sakata': 1,
    'Japan/Fukui': 2,
    'Japan/Toyama': 2,
    'Japan/Osaka': 4,
    'Japan/Obihiro': 0,
    'Japan/Kumamoto': 8,
    'Japan/Urakawa': 0,
    'Japan/Sendai-1': 1,
    'Japan/Shizuoka': 1,
    'Japan/Minamidaitojima': 10,
    'Japan/Yakushima-2': 9,
    'Japan/Takamatsu': 7,
    'Japan/Iriomotejima': 10,
    'Japan/Kyoto-1': 4,
    'Japan/Iida': 3,
    'Japan/Nara': 4,
    'Japan/Utsunomiya': 3,
    'Japan/Hamada': 5,
    'Japan/Yokohama': 3,
    'Japan/Shinjo': 1,
    'Japan/Iwamizawa': 0,
    'Japan/Sumoto': 4,
    'Japan/Hachijojima': 3,
    # 'Japan/Hachijojima': 10,  # TODO -- remove
    'Japan/Miyakojima': 10,
    'Japan/Nagoya-2': 6,
    'Japan/Matsumoto': 3,
    'Japan/Saigo': 3,
    'Japan/Tokyo': 3,
    'Japan/Aomori': 1,
    'Japan/Hamamatsu': 6,
    'Japan/Kofu': 3,
    'Japan/Tanegashima': 9,
    # 'Japan/Tanegashima': 10,  # TODO - - remove
    'Japan/Kochi-2': 7,
    'Japan/Sapporo': 0,
    'Japan/Yakushima-1': 9,
    'Japan/Tsu': 6,
    'Japan/Kumagaya': 3,
    'Japan/Hikone': 4,
    'Japan/Fukuoka': 8,
    'Japan/Mombetsu': 0,
    'Japan/Wakayama': 4,
    'Japan/Oshima': 3,
    'Japan/Owase': 6,
    'Japan/Kanazawa': 2,
    'Japan/Maebashi': 3,
    'Japan/Nobeoka': 9,
    'Japan/Morioka': 1,
    'Japan/Shirakawa': 6,
    'Japan/Akita': 1,
    'Japan/Izuhara-1': 8,
    'Japan/Uwajima': 7,
    'Japan/Naze,Funchatoge': 10,  # Officially classified as 9 -- climate is closer to 10
    'Japan/Choshi': 3,
    'Japan/Naha': 10,
    'Japan/Matsue': 5,
    'Japan/Takada': 2,
    'Japan/Yonago': 5,
    'Japan/Takayama': 3,
    'Japan/Hiroo': 0,
    'Japan/Aikawa': 2,
    'Japan/Hakodate': 0,
    'Japan/Tokushima': 7,
    'Japan/Hiroshima': 5,
    'Japan/Kumejima': 10,
    'Japan/Yonagunijima': 10,
    'Japan/Nagano': 3,
    'Japan/Miyazaki': 9,
    'Japan/Maizuru': 4,
    'Japan/Kutchan': 0,
}


VARIETIES = {
    0: 'someiyoshino',  # Prunus Yedoenis
    1: 'ezoyamazakura',  # Prunus sargentii
    2: 'hikanzakura',  # Prunus campanulata Maxim
    3: 'chishimazakura',  # Prunus nipponica Matsum
}

# Source: https://www.data.jma.go.jp/sakura/data/sakura004_07.html
LOCATION_VARIETY_JAPAN = {
    'Japan/Kushiro-1': 1,
    'Japan/Naze': 2,
    'Japan/Miyakejima': 0,
    'Japan/Fukushima': 0,
    'Japan/Kagoshima': 0,
    'Japan/Saga': 0,
    'Japan/Nagasaki': 0,
    'Japan/Muroran-2': 0,
    'Japan/Tottori-2': 0,
    'Japan/Niigata': 0,
    'Japan/Toyooka': 0,
    'Japan/Nagoya-1': 0,
    'Japan/Tateyama': 0,
    'Japan/Tottori-1': 0,
    'Japan/Nago': 2,
    'Japan/Kobe': 0,
    'Japan/Tsuruga': 0,
    'Japan/Shimonoseki': 0,
    'Japan/Miyako': 2,
    'Japan/Izuhara-2': 0,
    # 'Japan/Shionomisaki': ,
    'Japan/Asahikawa': 1,
    'Japan/Wakkanai': 1,
    'Japan/Fukue': 0,
    'Japan/Sendai-2': 0,
    'Japan/Mito': 0,
    'Japan/Nemuro': 3,
    'Japan/Wajima': 0,
    'Japan/Rumoi': 1,
    'Japan/Gifu': 0,
    'Japan/Hachinohe': 0,
    'Japan/Yamagata': 0,
    'Japan/Abashiri': 1,
    'Japan/Kyoto-2': 0,
    'Japan/Onahama': 0,
    'Japan/Matsuyama': 0,
    'Japan/Muroran-1': 0,
    'Japan/Oita': 0,
    # 'Japan/Ishigakijima': ,
    'Japan/Esashi': 0,
    'Japan/Kushiro-2': 1,
    'Japan/Kochi-1': 0,
    'Japan/Okayama': 0,
    'Japan/Sakata': 0,
    'Japan/Fukui': 0,
    'Japan/Toyama': 0,
    'Japan/Osaka': 0,
    'Japan/Obihiro': 1,
    'Japan/Kumamoto': 0,
    'Japan/Urakawa': 1,
    'Japan/Sendai-1': 0,
    'Japan/Shizuoka': 0,
    # 'Japan/Minamidaitojima': ,
    'Japan/Yakushima-2': 0,
    'Japan/Takamatsu': 0,
    # 'Japan/Iriomotejima': ,
    'Japan/Kyoto-1': 0,
    'Japan/Iida': 0,
    'Japan/Nara': 0,
    'Japan/Utsunomiya': 0,
    'Japan/Hamada': 0,
    'Japan/Yokohama': 0,
    'Japan/Shinjo': 0,
    'Japan/Iwamizawa': 1,
    'Japan/Sumoto': 0,
    'Japan/Hachijojima': 0,
    'Japan/Miyakojima': 2,
    'Japan/Nagoya-2': 0,
    'Japan/Matsumoto': 0,
    'Japan/Saigo': 0,
    'Japan/Tokyo': 0,
    'Japan/Aomori': 0,
    'Japan/Hamamatsu': 0,
    'Japan/Kofu': 0,
    'Japan/Tanegashima': 0,
    'Japan/Kochi-2': 0,
    'Japan/Sapporo': 0,
    'Japan/Yakushima-1': 0,
    'Japan/Tsu': 0,
    'Japan/Kumagaya': 0,
    'Japan/Hikone': 0,
    'Japan/Fukuoka': 0,
    # 'Japan/Mombetsu': ,
    'Japan/Wakayama': 0,
    'Japan/Oshima': 0,
    'Japan/Owase': 0,
    'Japan/Kanazawa': 0,
    'Japan/Maebashi': 0,
    'Japan/Nobeoka': 0,
    'Japan/Morioka': 0,
    'Japan/Shirakawa': 0,
    'Japan/Akita': 0,
    'Japan/Izuhara-1': 0,
    'Japan/Uwajima': 0,
    'Japan/Naze,Funchatoge': 2,
    'Japan/Choshi': 0,
    'Japan/Naha': 2,
    'Japan/Matsue': 0,
    'Japan/Takada': 0,
    'Japan/Yonago': 0,
    'Japan/Takayama': 0,
    'Japan/Hiroo': 1,
    'Japan/Aikawa': 0,
    'Japan/Hakodate': 0,
    'Japan/Tokushima': 0,
    'Japan/Hiroshima': 0,
    'Japan/Kumejima': 2,
    # 'Japan/Yonagunijima': ,
    'Japan/Nagano': 0,
    'Japan/Miyazaki': 0,
    'Japan/Maizuru': 0,
    # 'Japan/Kutchan': ,  # Kutchan observed Ezoyamazakura until 1994 and Someiyoshino from 1995 to 2006 .
}

LOCATION_VARIETY_SOUTH_KOREA = {  # TODO -- verify
    loc: 0 for loc in get_locations_south_korea()
}

LOCATION_VARIETY = {
    **LOCATIONS_REGIONS_JAPAN,
    **LOCATION_VARIETY_SOUTH_KOREA,
}

"""

    Predefined groups

    0: 'Hokkaido',
    1: 'Tohoku',
    2: 'Hokuriku',
    3: 'Kanto-Koshin',
    4: 'Kinki',
    5: 'Chugoku',
    6: 'Tokai',
    7: 'Shikoku',
    8: 'Kyushu-North',
    9: 'Kyushu-South-Amami',
    10: 'Okinawa',

"""

LOCATIONS_WO_OKINAWA = {k: v for k, v in LOCATIONS_REGIONS_JAPAN.items() if v != 10}

LOCATIONS_JAPAN_YEDOENIS = {k: v for k, v in LOCATION_VARIETY_JAPAN.items() if v == 0}
LOCATIONS_JAPAN_SARGENTII = {k: v for k, v in LOCATION_VARIETY_JAPAN.items() if v == 1}
LOCATIONS_JAPAN_CAMPANULATA = {k: v for k, v in LOCATION_VARIETY_JAPAN.items() if v == 2}
LOCATIONS_JAPAN_NIPPONICA = {k: v for k, v in LOCATION_VARIETY_JAPAN.items() if v == 3}

LOCATIONS_HOKKAIDO = {k: v for k, v in LOCATIONS_REGIONS_JAPAN.items() if v == 0}
LOCATIONS_TOHOKU = {k: v for k, v in LOCATIONS_REGIONS_JAPAN.items() if v == 1}
LOCATIONS_HOKURIKU = {k: v for k, v in LOCATIONS_REGIONS_JAPAN.items() if v == 2}
LOCATIONS_KANTO_KOSHIN = {k: v for k, v in LOCATIONS_REGIONS_JAPAN.items() if v == 3}
LOCATIONS_KINKI = {k: v for k, v in LOCATIONS_REGIONS_JAPAN.items() if v == 4}
LOCATIONS_CHUGOKU = {k: v for k, v in LOCATIONS_REGIONS_JAPAN.items() if v == 5}
LOCATIONS_TOKAI = {k: v for k, v in LOCATIONS_REGIONS_JAPAN.items() if v == 6}
LOCATIONS_SHIKOKU = {k: v for k, v in LOCATIONS_REGIONS_JAPAN.items() if v == 7}
LOCATIONS_KYUSHU_NORTH = {k: v for k, v in LOCATIONS_REGIONS_JAPAN.items() if v == 8}
LOCATIONS_KYUSHU_SOUTH_AMAMI = {k: v for k, v in LOCATIONS_REGIONS_JAPAN.items() if v == 9}
LOCATIONS_OKINAWA = {k: v for k, v in LOCATIONS_REGIONS_JAPAN.items() if v == 10}


if __name__ == '__main__':
    from evaluation.plots.maps import savefig_location_annotations_on_map

    _locations = list(LOCATION_VARIETY_JAPAN.keys())
    _cmap = {
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'purple',
    }
    _colors = [_cmap[LOCATION_VARIETY_JAPAN[_loc]] for _loc in _locations]

    savefig_location_annotations_on_map(
        # annotations=[''] * len(_locations),
        [_loc.split('/')[1] for _loc in _locations],
        locations=_locations,
        path='temp',
        colors=_colors,
    )


