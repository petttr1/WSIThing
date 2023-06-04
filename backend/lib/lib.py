from lib.Utils.storage import Storage
from lib.Utils.fe_wsi_maker import FeWSIMaker
from lib.Parsers.xml_parser import XMLParser


def create_storage():
    return Storage()


def create_fe_analyzer(path, name, parser=None):
    return FeWSIMaker(path, name, parser)


def transform_to_d3(array):
    xs, ys = [], []
    for el in array:
        stripped = el[0]
        xs.append(int(stripped[0]))
        ys.append(int(stripped[1]))
    return {'x': xs, 'y': ys}


def create_annotation(annotation):
    # colors = ['red', 'green', 'blue', 'cyan', 'fuchsia', 'aqua', 'yellow']
    parser = XMLParser(annotation)
    regions = parser.parse_annotation()
    region_types = list(set(regions.keys()))
    if len(region_types) > 7:
        return False
    polygons = []
    for i, rt in enumerate(region_types):
        for region_type, value in regions.items():
            if region_type == rt:
                for region in value['region_vertices']:
                    r = {
                        'points': transform_to_d3(region),
                    }
                    polygons.append(r)
    return polygons, parser, region_types
