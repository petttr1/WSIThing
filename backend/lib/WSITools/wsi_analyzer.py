from sys import path
import openslide as ops
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import box, Polygon
from shapely.strtree import STRtree
from PIL import Image


from lib.WSITools.batch_producer import ParametrizedBatchProducer
from lib.WSITools.quad_bunch import QuadBunch


# noinspection PyInterpreter
class WSIAnalyzer:
    wsi_quad = None

    def __init__(self, path_to_slide, annotation_parser=None, quad_bunch_json_export=None, classes_contained=[], specific_roots=[], bad_quality_function=None, wsi=None):
        """Class for analyzing histopathology WSIs.

        Args:
            path_to_slide (str): path to the slide to be analyzed
            annotation_parser (str, optional): path to the annotation associated with the slide. Defaults to None.
            quad_bunch_json_export (dict, optional): export of previous analyses performed with the WSI. Defaults to None.
            classes_contained (list, optional): list of classes contained within the annotation as a list of str. Defaults to [].
            specific_roots (list, optional): int indices of specific quad bunch roots to use. Defaults to [].
            bad_quality_function (function, optional): the function to use to decide whether a region is bad quality. Defaults to None.
        """
        self.path_to_slide = path_to_slide
        self.specific_roots = specific_roots
        self.quad_bunch_json_export = quad_bunch_json_export
        self.annotation_parser = annotation_parser
        self.bad_quality_function = bad_quality_function
        self.batch_producers = {}

        if classes_contained == []:
            if annotation_parser != None:
                self.classes_contained = self.annotation_parser.regions
        else:
            self.classes_contained = classes_contained

        if wsi != None:
            self.wsi = wsi
        else:
            self.wsi = self.__load_slide()

        if self.annotation_parser != None:
            self.regions = self.annotation_parser.parse_annotation()
            self.__build_rtree()
            # self.mask = self.create_mask_from_regions(
            #     img_shape=self.wsi.level_dimensions[1][::-1])
            self.assessments = self.__assess_using_quad_bunch(True)
            self.cancer_types = list(set(self.assessments['type']))
            self.cancer_types_count = len(self.cancer_types)
            self.classes = {tumor_type: one_hot for one_hot,
                            tumor_type in enumerate(self.cancer_types)}
        else:
            self.assessments = self.__assess_using_quad_bunch(False)

    def extended_setup(self):
        pass

########## ANALYZER EXPOSED METHODS ##########
    def get_analysis_for_expert(self, expert, downscale=16, thresh=0.6, weights=None):
        # TODO: hard-coded downscale factor, number of classes
        return self.make_polygons_from_mask(self.wsi_quad.get_mask(downscale, expert, True, False), downscale, thresh, self.wsi_quad.get_class_ordering(expert), weights)

    def get_analysis_for_classes(self, classes):
        analyses = {}
        for c in classes:
            # TODO: hard-coded down, thresh
            analyses[c] = self.make_polygons_from_mask(
                self.create_mask_from_regions(tumor_types=[c]), 16, 0.6, classes)
        return analyses

    def transform_to_d3(self, poly, downscale):
        res = []
        xs, ys = [], []
        try:
            if not poly.is_empty:
                xs, ys = poly.exterior.xy
                res.append({'x': list(int(x * downscale) for x in xs),
                            'y': list(int(y * downscale) for y in ys)})
        except:
            for geom in poly.geoms:
                xx, yy = geom.exterior.xy
                res.append({'x': list(int(x * downscale) for x in xx),
                            'y': list(int(y * downscale) for y in yy)})
        return res

    def make_polygons_from_mask(self, img, downscale, thresh, classes, weights=None):
        colors = ['red', 'green', 'blue', 'cyan', 'fuchsia', 'aqua', 'yellow']
        polys = []
        if weights:
            print('weights', weights)
            img = img * np.array(weights).astype(np.float32)
            img_max = np.max(img)
            img_min = np.min(img)
            img = (img - img_min) / (img_max - img_min)
        for channel in range(img.shape[-1]):
            im = (255*(img[:, :, channel] > thresh)).astype(np.uint8)
            if np.average(im) == 255 or np.sum(im) == 0:
                continue
            contours, hierarchy = cv2.findContours(
                image=im, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            # RETR_TREE
            for v in contours:
                points = np.squeeze(v, 1)
                p = Polygon(points)
                p = p.buffer(0)
                for individual_poly in self.transform_to_d3(p, downscale):
                    polys.append({
                        'points': individual_poly,
                        'class': classes[channel],
                        'fill': colors[channel],
                    })
        return polys

    def set_classes(self, new_classes):
        """Sets its own classes with *new_classes*. Used when analyzing multiple WSIs with each containing different number of classes.

            Args:
                new_classes (dict): classes to set as this analyzers' classes
        """
        self.classes = new_classes

    def get_levels(self):
        """Returns the dimensions of all the native levels of the WSI.

        Returns:
            tuple[tuple[int, int], ...]: the downsample levels
        """
        return self.wsi.level_dimensions

    def get_wsi(self, down_level):
        """Returns WSI downsampled to *down_level* level.

        Args:
            down_level (int): The downsize level desired. Dimensions of all downsize levels may be listed using the get_levels() method.

        Returns:
            PIL.Image: the downsized WSI
        """
        return self.wsi.get_thumbnail(self.wsi.level_dimensions[down_level])

    def create_mask_from_regions(self, img_shape=None, tumor_types=[]):
        """Creates segmentation mask for select tumor types from WSI and returns it.
        """
        if img_shape == None:
            img_shape = self.wsi.level_dimensions[-1][::-1]
        zoom_level = round(self.wsi.level_dimensions[0][0] / img_shape[1])
        img = np.zeros(img_shape, dtype=np.uint8)
        for region_type, value in self.regions.items():
            if len(tumor_types) == 0 or region_type in tumor_types:
                for region in value['region_vertices']:
                    cv2.fillPoly(img, np.array(
                        [region // zoom_level], dtype=np.int32), 255)
        return img

    def get_analysis_mask(self, analyzer, downscale_factor=16, exclude_bad_quality=True, zoom_levels=[]):
        """Constructs and returns an analysis produced by the analyzer in the form of an image.

        Args:
            analyzer (str): which analyzer to produce mask for
            downscale_factor (int, optional): factor by which to downscale the WSI shape when creating the mask. Defaults to 16.
            exclude_bad_quality (bool, optional): whether to exclude the regions identified as bad quality when producing the mask (the regions will have the value of 0). Defaults to True.
            zoom_levels (list, optional): which zoom levels should the mas contain. Empty list means all zoom levels. Defaults to [].

        Returns:
            Image: the mask created by combining all of the analyses for a specific analyzer.
        """
        return self.wsi_quad.get_mask(downscale_factor, analyzer, len(list(self.classes.keys())), exclude_bad_quality=exclude_bad_quality, use_weights=False, weighing_name="", zoom_levels=zoom_levels)

    def prepare_samples(self, type, size, eval_data=False):
        """Returns samples and their annotations.

            Args:
                type (str): the tumor type for which to produce images
                size (int): size of one side of the final image

            Returns:
                (np.array, np.array): the batch consisting of images and labels
        """
        images, labels = [], []
        # get batch as records in a DF
        batch_df = self.batch_producers[type].produce_batch(eval_data)
        for _, (def_id,
                id,
                x,
                y,
                width,
                height,
                depth,
                tum_type,
                quality,
                damaged_area) in enumerate(batch_df.itertuples()):
            # read, resize each img in a batch
            window = self.wsi.read_region(
                (x, y), 0, (width, height))
            window = window.resize(
                (size, size), Image.ANTIALIAS)
            images.append(self.__prepare_image(window) / 255.)
            labels.append(type)
        return images, labels

########## ANALYZER UTILS ##########

    def __load_slide(self):
        """Loads the slide from file and returns the reference.

        Returns:
            ops.OpenSlide: the reference top the loaded slide
        """
        try:
            slide = ops.OpenSlide(self.path_to_slide)
        except:
            raise
        return slide

    def __build_rtree(self):
        """Builds the RTree using the annotation.
        """
        global_regions = []
        for _, region in self.regions.items():
            global_regions.extend(region['region_polygons'])
        self.rtree = STRtree(global_regions)

    def assert_type_from_rtree(self, x, y, width, height):
        """Asserts the type of a region from the annotation.

        Args:
            x (int): the x coordinate of the assessed region
            y (int): the y coordinate of the assessed region
            width (int): width of the region
            height (int): height of the region

        Returns:
            (str, float): (window type, area of the region which is damaged by the type)
        """
        current_window = box(x, y,
                             x + width,
                             y + height)
        window_type = 'Normal'
        # get the region which are contained within current window
        contained_regions = [r for r in self.rtree.query(
            current_window) if r.intersects(current_window)]
        contained_areas = {t: 0 for t in self.classes_contained}
        # for each region:
        for r in contained_regions:
            region_area = current_window.intersection(r).area
            for tumor_type, region in self.regions.items():
                # if the current region r is in regions of a type, we can assess the window as this type
                if r in region['region_polygons']:
                    contained_areas[tumor_type] += region_area / \
                        current_window.area
        contained_areas['Normal'] = 1 - sum(contained_areas.values())
        window_type = max(contained_areas, key=contained_areas.get)
        return window_type, contained_areas

    def __assess_using_quad_bunch(self, use_type_function):
        """Assesses WSI and creates DataFrame of windows + annotations.

        Args:
            use_type_function (bool): whether to use a type funkction when assessing the WSI

        Returns:
            pd.DataFrame: [description]
        """
        if self.quad_bunch_json_export != None:
            self.wsi_quad = QuadBunch(
                self.wsi, use_json=True, json_export=self.quad_bunch_json_export)
            self.available_analyses = self.wsi_quad.collect_available_analyses()
            print('ANALYSES:', self.available_analyses)
            return pd.DataFrame(self.wsi_quad.get_assessments(self.bad_quality_function, None))
        else:
            if self.annotation_parser != None:
                self.wsi_quad = QuadBunch(
                    self.wsi, default_assessment_function=self.assert_type_from_rtree, roots_to_use=self.specific_roots)
            else:
                self.wsi_quad = QuadBunch(self.wsi)
        return pd.DataFrame(self.wsi_quad.get_assessments(self.bad_quality_function, self.assert_type_from_rtree if use_type_function else None))

    def register_batch_producers(self, types, zoom_levels, quality_options, batch_size):
        """Register the batch producers making them usable for selecting the data to be generated.

        Args:
            types (dict): Types of tumors. The mapping is {'name':[types]}, so one "class" may contain multiple tumor types
            zoom_levels (list): zoom levels to use with the batch producer
            quality_options (list): quality options to use
            batch_size (int): batch size to use
        """
        self.batch_producers = {}
        for type_name, types in types.items():
            df = self.assessments[(self.assessments['type'].isin(types))
                                  & (self.assessments['quality'].isin(quality_options))
                                  & (self.assessments['depth'].isin(zoom_levels))]
            self.batch_producers[type_name] = ParametrizedBatchProducer(
                df, batch_size)

    def __prepare_image(self, window):
        """Prepares the image. Converts the PIL Image to np array representing the image.

        Args:
            window (PIL.Image): the image to be prepared

        Returns:
            np.array: prepared image
        """
        # 4 to 3 channels hack
        color = (255, 255, 255)
        image = Image.new('RGB', window.size, color)
        # 3 is the alpha channel
        image.paste(window, mask=window.split()[3])
        return np.array(image)
