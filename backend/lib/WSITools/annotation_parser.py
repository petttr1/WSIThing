from abc import ABC, abstractmethod
from shapely.geometry import Polygon


class AnnotationParser(ABC):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.regions = []

    def parse_annotation(self):
        """Constructs polygons of type *shapely.geometry.Polygon* and adds them to the dict of vertices. 
        The Analyzer uses the Polygons to work with the annotation.

        Returns:
            dict: dict of the regions extracted
        """
        regions_dict = self.extract_vertices()
        self.regions = list(set(regions_dict.keys()))
        for region, item in regions_dict.items():
            polys = []
            for v in item['region_polygons']:
                p = Polygon(v)
                p = p.buffer(0)
                polys.append(p)
            regions_dict[region]['region_polygons'] = polys
        return regions_dict

    @abstractmethod
    def extract_vertices(self):
        """Parses the annotation, constructs and returns a dict following the template:
            {
                'tumor_type':
                {
                    'region_vertices': [np arrays of vertices for different regions of type *tumor_type*],
                }
            }
        """
        pass
