import math
import numpy as np
import cv2


class Quad:
    q1 = None
    q2 = None
    q3 = None
    q4 = None
    size = (None, None)
    depth = 0
    is_bad_quality = False
    idx = None
    tumor_type = None
    damaged_amount = None

    def __init__(self, root, idx, coordinates, size, depth, granularity, parents, construct_from_json=False, json_template={}):
        """The quad used to store info about a certain part of the WSI.

        Args:
            root (Quad): Root of the current t-pyramid
            idx (int): id of the current quad
            coordinates ((int,int)): TOP LEFT coordinates of this quad
            size ((int, int)): width and length of this quad
            depth (int): depth of the current node
            granularity (int): minimum size of one side
            parents (list): ids of all parents of the current node
            construct_from_json (bool, optional): Whether the quad is loaded from a JSON. If True, the JSON needs to be one of the arguments. Defaults to False.
            json_template (dict, optional): JSON to use to construct the quad. Defaults to {}.
        """
        if construct_from_json == True:
            self.json_construct(json_template)
        else:
            self.root = root
            self.idx = idx
            self.parents = parents
            self.y, self.x = coordinates
            self.height, self.width = size
            self.granularity = granularity
            self.depth = depth
            self.analyses = {}
            if self.width > self.granularity and self.height > self.granularity:
                width = math.ceil(self.width / 2)
                height = math.ceil(self.height / 2)
                self.q1 = Quad(
                    self.root,
                    self.idx * 4 + 1,
                    (self.y, self.x),
                    (height, width),
                    self.depth + 1,
                    self.granularity,
                    self.parents + [self.idx, ])
                self.q2 = Quad(
                    self.root,
                    self.idx * 4 + 2,
                    (self.y, self.x + width),
                    (height, width),
                    self.depth + 1,
                    self.granularity,
                    self.parents + [self.idx, ])
                self.q4 = Quad(
                    self.root,
                    self.idx * 4 + 3,
                    (self.y + height, self.x),
                    (height, width),
                    self.depth + 1,
                    self.granularity,
                    self.parents + [self.idx, ])
                self.q3 = Quad(
                    self.root,
                    self.idx * 4 + 4,
                    (self.y + height, self.x + width),
                    (height, width),
                    self.depth + 1,
                    self.granularity,
                    self.parents + [self.idx, ])

    def set_analysis_for_analyzer(self, name, analysis):
        """Stores the analysis for a specific analyzer.

        Args:
            name (str): the name for the analyzer
            analysis (any): the analysis to be stored
        """
        self.analyses[name] = analysis

    def get_analysis_for_analyzer(self, name, shape, num_classes, exclude_bad_quality=True):
        """Returns an analysis for a specific analyzer.

        Args:
            name (str): name of the analyzer
            shape (tuple): shape of the resulting analysis mimicking an analysis for an image
            num_classes (int): number of classes included in the result
            exclude_bad_quality (bool, default=True): whether to exclude bad quality regions of the WSI

        Returns:
            np.array: the analysis in the shape of a 3d array of values
        """
        img = np.full((shape[0], shape[1], num_classes), 0.)
        if self.is_bad_quality and exclude_bad_quality:
            return img
        for i, val in enumerate(self.analyses[name]):
            if hasattr(val, "__len__"):
                img[i] = cv2.resize(
                    val, dsize=shape, interpolation=cv2.INTER_AREA)
            else:
                img[:, :, i] = np.full(shape, val)
        return img

    def set_bad_quality(self):
        """Sets the current quad as bad quality.
        """
        self.is_bad_quality = True

    def get_is_bad_quality(self):
        """Check whether the current quad is bad quality.

        Returns:
            bool: True if the current quad is bad quality, False otherwise
        """
        return self.is_bad_quality

    def set_type(self, tumor_type, damaged_amount):
        """Sets the type and damaged amount for this quad.

        Args:
            type (str): tumor type of this region.
            damaged_amount (float): the area of this region containing the tumor.
        """
        self.tumor_type = tumor_type
        self.damaged_amount = damaged_amount

    def get_children(self):
        """Returns all the children of the current quad if any are present.

        Returns:
            list: 4 children of the current quad if they exist, else empty list
        """
        if self.q1 is not None and self.q2 is not None and self.q3 is not None and self.q4 is not None:
            return [self.q1, self.q2, self.q3, self.q4]
        else:
            return []

    def add_node_to_json(self):
        """Adds this quad to the JSON being exported.

        Returns:
            dict: all info about the current quad
        """
        if self.q1 is not None and self.q2 is not None and self.q3 is not None and self.q4 is not None:
            children = [self.q1.add_node_to_json(), self.q2.add_node_to_json(
            ), self.q3.add_node_to_json(), self.q4.add_node_to_json()]
        else:
            children = []
        params = {
            'root': self.root,
            'idx': self.idx,
            'x': self.x,
            'y': self.y,
            'height': self.height,
            'width': self.width,
            'granularity': self.granularity,
            'parents': self.parents,
            'depth': self.depth,
            'bad_quality': self.is_bad_quality,
            'type': self.tumor_type,
            'damaged_amount': self.damaged_amount,
            'analyses': self.analyses,
            'children': children
        }
        return params

    def json_construct(self, json):
        """Transfers the info from JSON to the current quad.

        Args:
            json (dict): the info about the current quad in the JSON format
        """
        self.root = json['root']
        self.idx = json['idx']
        self.x = json['x']
        self.y = json['y']
        self.height = json['height']
        self.width = json['width']
        self.granularity = json['granularity']
        self.depth = json['depth']
        self.is_bad_quality = json['bad_quality']
        self.analyses = json['analyses']
        self.parents = json['parents']
        self.tumor_type = json['type']
        self.damaged_amount = json['damaged_amount']
        if json['children']:
            self.q1 = Quad(None, None, None, None, None, None,
                           None, True, json['children'][0])
            self.q2 = Quad(None, None, None, None, None, None,
                           None, True, json['children'][1])
            self.q3 = Quad(None, None, None, None, None, None,
                           None, True, json['children'][2])
            self.q4 = Quad(None, None, None, None, None, None,
                           None, True, json['children'][3])
        else:
            self.q1 = None
            self.q2 = None
            self.q3 = None
            self.q4 = None
