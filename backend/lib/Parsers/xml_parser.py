import numpy as np
import xml.etree.ElementTree as ET

from lib.WSITools.annotation_parser import AnnotationParser


class XMLParser(AnnotationParser):
    def __init__(self, path):
        super().__init__(path)

    def extract_vertices(self):
        """Loads and prepares the annotation."""
        # xml tree reading
        tree = ET.parse(self.path)
        root = tree.getroot()
        regions_dict = {}
        # iterate over annotated regions of the image
        for region in root.find('Annotation').find('Regions').findall('Region'):
            vertices = []
            # find all vertices and add them to the region
            for vertex in region.find('Vertices'):
                vertices.append((int(round(float(vertex.attrib['X']))),
                                 int(round(float(vertex.attrib['Y'])))))
            # store the vertices in the dict
            try:
                regions_dict[region.attrib['Text']]['region_vertices']\
                    .append(np.array(vertices).reshape((-1, 1, 2)).astype(np.int32))
                regions_dict[region.attrib['Text']
                             ]['region_polygons'].append(vertices + [vertices[-1]])
            except KeyError:
                regions_dict[region.attrib['Text']] = {
                    'region_vertices': [np.array(vertices).reshape((-1, 1, 2)).astype(np.int32)],
                    'region_polygons': [vertices + [vertices[-1]]]
                }
        return regions_dict
