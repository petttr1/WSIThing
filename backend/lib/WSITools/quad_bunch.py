import math
from os import stat
import numpy as np
import sys
import cv2
from PIL import Image
# from multiprocessing import Array
# from tqdm import tqdm

from lib.WSITools.queue_processing import WSIQueue
from lib.WSITools.quad import Quad


class QuadBunch:
    def __init__(self, image, root_size=8192, granularity=256, use_json=False, json_export={}, default_assessment_function=None, roots_to_use=[]):
        """A structure used to keep track of all the quads within a WSI.

        Args:
            image (ops.OpenSlide): WSI to be divided into quads
            root_size (int, optional): size of the roots. Defaults to 8192.
            granularity (int, optional): division stopping size. Defaults to 256.
            use_json (bool, optional): whether to use a JSON export to load data from. Defaults to False.
            json_export (dict, optional): JSON export to load data from. Defaults to {}.
            default_assessment_function (function, optional): a function to create a default assessment (e.g. read from annotation). Defaults to None.
            roots_to_use (list, optional): indices of specific roots to use. Defaults to [].
        """
        self.image = image
        self.use_json = use_json
        self.json_export = json_export
        self.roots_to_use = roots_to_use
        if self.use_json == True:
            try:
                self.analyses_classes_ordering = self.json_export['analyses_classes_ordering']
            except:
                self.analyses_classes_ordering = {'gt': ['Benign', 'Carcinoma in situ', 'Invasive carcinoma', 'Normal'], 'run_4_aug': [
                    'Benign', 'Carcinoma in situ', 'Invasive carcinoma', 'Normal']}
            self.quad_root_size = self.json_export['root_size']
            self.max_depth = self.json_export['max_depth']
            self.granularity = self.json_export['granularity']
            self.roots = self.construct_from_json()
        else:
            self.analyses_classes_ordering = {}
            self.granularity = granularity
            self.quad_root_size = root_size
            self.max_depth = self.__find_max_depth()
            self.roots = self.__create_quads()
        if default_assessment_function is not None:
            self.apply_default_analysis(
                default_assessment_function)

    def get_class_ordering(self, analyzer_name):
        return self.analyses_classes_ordering[analyzer_name]

    @staticmethod
    def default_analysis_callback(args):
        node = args[0]
        function = args[1]
        gt_values = np.array(list(function(node.x, node.y, node.width, node.height)[
            1].values()))
        node.set_analysis_for_analyzer(
            'gt', np.where(gt_values == gt_values.max(), 1., 0.))

    def apply_default_analysis(self, function):
        """Applies the default analysis in a form of *function* to all quads contained.

        Args:
            function (function): the function to apply
        """
        queue = WSIQueue()
        nodes = queue.extract_nodes(self.roots)
        for node in nodes:
            self.default_analysis_callback((node, function))
        # queue.process_nodes(
        #     list(zip(nodes, list(function for _ in range(len(nodes))))), self.default_analysis_callback)
        # for root in self.roots:
        #     queue = [root]
        #     while queue:
        #         node = queue.pop(0)
        #         node_children = node.get_children()
        #         queue.extend(node_children)
        #         gt_values = np.array(list(function(node.x, node.y, node.width, node.height)[
        #             1].values()))
        #         node.set_analysis_for_analyzer(
        #             'gt', np.where(gt_values == gt_values.max(), 1., 0.))

    def get_window(self, node, root, region, size):
        """Gets a window from a larger region.

        Args:
            node (Quad): node containing the region
            root (Quad): root of the t-pyramid
            region (np.array): image to read from
            size (int): size of one side of the resulting window

        Returns:
            np.array: the selected window
        """
        x = node.x - root.x
        y = node.y - root.y
        w = node.width
        h = node.height
        window = region[y:y+h, x:x+w]
        pil_window = Image.fromarray(window)
        pil_resized = pil_window.resize(
            (size, size), Image.ANTIALIAS)
        window_np = np.array(pil_resized) / 255.
        return window_np

    def apply_weighing(self, analyzer):
        """Applies weighing to the analyses.
        """
        models = analyzer.models
        input_shape = (analyzer.input_shape[0], analyzer.input_shape[1])
        for root in self.roots:
            # Create a dict for saving predicted nodes of previous levels
            pred_nodes = {}
            # Load the root
            analyzed_region = self.image.read_region(
                (root.x, root.y), 0, (root.width, root.height))
            color = (255, 255, 255)
            image = Image.new('RGB', analyzed_region.size, color)
            image.paste(analyzed_region, mask=analyzed_region.split()[3])
            image = np.array(image)

            # get all the quads and predict values for them
            info = {
                'batches': {k: [] for k in range(6)},
                'indices': {k: [] for k in range(6)}
            }
            queue = [root]
            while queue:
                node = queue.pop(0)
                node_children = node.get_children()
                queue.extend(node_children)
                info['batches'][node.depth].append(
                    self.get_window(node, root, image, input_shape[0]))
                info['indices'][node.depth].append(node.idx)

            for i in range(6):
                np_batch = np.array(info['batches'][i])
                pred = models[i].predict(np_batch, batch_size=512)
                for pr, idx in zip(pred, info['indices'][i]):
                    pred_nodes[idx] = pr

            queue = [root]
            while queue:
                node = queue.pop(0)
                node_children = node.get_children()
                queue.extend(node_children)
                if node.depth == 5:
                    vals = [pred_nodes[node.idx][0]]
                    for parent_idx in node.parents[::-1]:
                        vals.append(pred_nodes[parent_idx][0])
                    vals = np.array(vals)
                    node.set_analysis_for_analyzer(analyzer.name, vals)

    def __find_max_depth(self):
        """Finds the maximum depth of the t-pyramids

        Returns:
            int: the maximum depth
        """
        size = self.quad_root_size
        depth = 0
        while(size > self.granularity):
            size /= 2
            depth += 1
        return depth

    def __create_quads(self):
        """Creates roots for the t-pramids and the t-pyramids, dividing the WSI into Quads and organising them.

        Returns:
            list: the created roots
        """
        width = self.image.level_dimensions[0][0]
        height = self.image.level_dimensions[0][1]
        img_ratio = width / height

        cols = width / self.quad_root_size
        rows = cols / img_ratio
        cols = math.ceil(cols)
        rows = math.ceil(rows)

        offset_w = math.ceil(2*(width / cols) - self.quad_root_size)
        offset_h = math.ceil(2*(height / rows) - self.quad_root_size)

        roots = []
        root_idx = 0
        for i in range(0, width - offset_w, offset_w):
            for j in range(0, height - offset_h, offset_h):
                if root_idx in self.roots_to_use or not self.roots_to_use:
                    start_width = i
                    start_height = j
                    if start_width + self.quad_root_size > width:
                        start_width = width - self.quad_root_size
                    if start_height + self.quad_root_size > height:
                        start_height = height - self.quad_root_size
                    roots.append(
                        Quad(
                            root_idx,
                            0,
                            (start_height, start_width),
                            (self.quad_root_size, self.quad_root_size),
                            0,
                            self.granularity,
                            []
                        )
                    )
                root_idx += 1
        return roots

    def export_json(self):
        """Exports all the info about the current WSI as a JSON.

        Returns:
            dict: the export of all the info
        """
        export = {
            'granularity': self.granularity,
            'root_size': self.quad_root_size,
            'max_depth': self.max_depth,
            'roots': []
        }
        for root in self.roots:
            export['roots'].append(root.add_node_to_json())
        return export

    def construct_from_json(self):
        """Loads the info from a JSON and constructs a QuadBunch with all the quads for a WSI

        Returns:
            list: roots
        """
        roots = [Quad(None, None, None, None, None, None, None, True, root)
                 for root in self.json_export['roots']]
        return roots

    @staticmethod
    def analyses_callback(node):
        return list(node.analyses.keys())

    def collect_available_analyses(self):
        queue = WSIQueue()
        nodes = queue.extract_nodes(self.roots)
        analyses = list(map(lambda x: self.analyses_callback(x), nodes))
        # analyses = queue.process_nodes(nodes, self.analyses_callback)
        # print('ANALYSES', list(
        #     set([item for sublist in analyses for item in sublist])))
        return list(set([item for sublist in analyses for item in sublist]))
        # analyses = []
        # for root in self.roots:
        #     queue.append(root)
        #     while queue:
        #         node = queue.pop(0)
        #         queue.extend(node.get_children())
        #         for analysis in list(node.analyses.keys()):
        #             if analysis not in analyses:
        #                 analyses.append(analysis)
        # return analyses

    def find_number_of_channels(self, root, analyzer):
        node = root
        analysis = None
        while analysis is None:
            try:
                analysis = node.analyses[analyzer]
                return analysis.shape[-1]
            except:
                pass

    @staticmethod
    def mask_callback(args):
        node = args[0]
        analyzer_name = args[1]
        root = args[2]
        channels = args[3]
        exclude_bad_quality = args[4]
        downscale_factor = args[5]
        root_mask = args[6]

        print('NODE ID: ', node.id)

        shape = (node.width//downscale_factor,
                 node.height//downscale_factor)

        x = node.x - root.x
        y = node.y - root.y

        # m is reference to an area within the constructed mask
        m = root_mask[node.depth, y // downscale_factor: y // downscale_factor + node.height // downscale_factor,
                      x // downscale_factor: x // downscale_factor + node.width // downscale_factor]
        # analysis is a scalar/vector of predicted classes
        analysis = node.get_analysis_for_analyzer(
            analyzer_name, shape, channels, exclude_bad_quality)
        # assign the region m is referencing the *analysis* values
        m = analysis
        root_mask[node.depth, y // downscale_factor: y // downscale_factor + node.height // downscale_factor,
                  x // downscale_factor: x // downscale_factor + node.width // downscale_factor] = m
        # if node.depth == 5 and use_weights:
        #     weights = node.get_analysis_for_analyzer(
        #         weighing_name, shape, 6, exclude_bad_quality)
        #     weights = np.swapaxes(weights[np.newaxis, :], 0, -1)

        #     root_mask[::-1, y // downscale_factor: y // downscale_factor + node.height // downscale_factor,
        #               x // downscale_factor: x // downscale_factor + node.width // downscale_factor] *= weights
        #     minmax = root_mask[:, y // downscale_factor: y // downscale_factor + node.height // downscale_factor,
        #                        x // downscale_factor: x // downscale_factor + node.width // downscale_factor]
        #     minmax = (minmax - minmax.min()) / \
        #         (minmax.max() - minmax.min() + 1e-7)
        #     root_mask[:, y // downscale_factor: y // downscale_factor + node.height // downscale_factor,
        #               x // downscale_factor: x // downscale_factor + node.width // downscale_factor] = minmax

    @staticmethod
    def weighted_mask_callback(args):
        pass

    def get_mask(self, downscale_factor, analyzer_name, exclude_bad_quality=True, use_weights=True, weighing_name="", zoom_levels=[]):
        """Constructs a mask of analyses, from all quads contained, with values <0, 1> for each pixel

        Args:
            downscale_factor (int): a factor with which to scale down the resulting mask
            analyzer_name (str): a name of the analyzer for which to construct the mask of analyses
            exclude_bad_quality (bool, optional): Whether to exclude bad quality regions. Defaults to True.
            use_weights (bool, optional): Whether to use weighing. Defaults to True.
            weighing_name (str, optional): Name of the weighing analysis. Defaults to "".
            zoom_levels (list, optional): zoom levels to use. Defaults to [].

        Returns:
            [type]: [description]
        """
        channels = self.find_number_of_channels(self.roots[0], analyzer_name)
        mask = np.zeros(
            (self.image.level_dimensions[0][1] // downscale_factor, self.image.level_dimensions[0][0] // downscale_factor, channels), dtype=np.float32)
        for i, root in enumerate(self.roots):
            queue = WSIQueue()
            nodes = list(filter(
                lambda node: node.depth in zoom_levels or not zoom_levels, queue.extract_nodes([root])))
            root_mask = np.zeros(
                (6, root.width // downscale_factor, root.height // downscale_factor, channels), dtype=np.float32)

            # if use_weights:
            #     queue.process_nodes(nodes, self.weighted_mask_callback)
            # else:
            #     print('SPAWNING PROCESSES')
            #     queue.process_nodes(list(zip(
            #         nodes, an_names, roots, chs, bad_q, down_factor, root_masks)), self.mask_callback)
            #     # , {'arr': X, 'size': rmask_shape[0] * rmask_shape[1] * rmask_shape[2]})
            for node in nodes:
                shape = (node.width//downscale_factor,
                         node.height//downscale_factor)

                x = node.x - root.x
                y = node.y - root.y

                # m is reference to an area within the constructed mask
                m = root_mask[node.depth, y // downscale_factor: y // downscale_factor + node.height // downscale_factor,
                              x // downscale_factor: x // downscale_factor + node.width // downscale_factor]
                # analysis is a scalar/vector of predicted classes
                analysis = node.get_analysis_for_analyzer(
                    analyzer_name, shape, channels, exclude_bad_quality)
                # assign the region m is referencing the *analysis* values
                m = analysis
                root_mask[node.depth, y // downscale_factor: y // downscale_factor + node.height // downscale_factor,
                          x // downscale_factor: x // downscale_factor + node.width // downscale_factor] = m
                if node.depth == 5 and use_weights:
                    weights = node.get_analysis_for_analyzer(
                        weighing_name, shape, 6, exclude_bad_quality)
                    weights = np.swapaxes(weights[np.newaxis, :], 0, -1)

                    root_mask[::-1, y // downscale_factor: y // downscale_factor + node.height // downscale_factor,
                              x // downscale_factor: x // downscale_factor + node.width // downscale_factor] *= weights
                    minmax = root_mask[:, y // downscale_factor: y // downscale_factor + node.height // downscale_factor,
                                       x // downscale_factor: x // downscale_factor + node.width // downscale_factor]
                    minmax = (minmax - minmax.min()) / \
                        (minmax.max() - minmax.min() + 1e-7)
                    root_mask[:, y // downscale_factor: y // downscale_factor + node.height // downscale_factor,
                              x // downscale_factor: x // downscale_factor + node.width // downscale_factor] = minmax
            root_mask = np.sum(root_mask, axis=0)
            root_mask = (root_mask - root_mask.min()) / \
                (root_mask.max() - root_mask.min() + 1e-7)

            root_area = mask[root.y // downscale_factor: root.y // downscale_factor + root.height // downscale_factor,
                             root.x // downscale_factor: root.x // downscale_factor + root.width // downscale_factor]
            root_area = np.where(
                root_area == 0,
                root_mask,
                (root_area + root_mask)/2
            )
            mask[root.y // downscale_factor: root.y // downscale_factor + root.height // downscale_factor,
                 root.x // downscale_factor: root.x // downscale_factor + root.width // downscale_factor] = root_area
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        return mask

    def get_assessments(self, quality_func, type_function):
        """For each window assesses the windows' type, damaged stroma amount and quality.

        Args:
            quality_func (function): a function to decide the quality of a region
            type_function (function): a function to decide the type of a region

        Returns:
            dict: dict with all info about the slide
        """

        slide_dict = {
            'id': [],
            'x_coord': [],
            'y_coord': [],
            'width': [],
            'height': [],
            'depth': [],
            'type': [],
            'quality': [],
            'damaged_amount': [],
        }
        if self.use_json:
            for root in self.roots:
                queue = [root]
            while queue:
                node = queue.pop(0)
                queue.extend(node.get_children())
                try:
                    slide_dict['type'].append(node.type)
                    slide_dict['damaged_amount'].append(node.damaged_amount)
                except:
                    slide_dict['type'].append('nan')
                    slide_dict['damaged_amount'].append(-1)
                slide_dict['x_coord'].append(node.x)
                slide_dict['y_coord'].append(node.y)
                slide_dict['width'].append(node.width)
                slide_dict['height'].append(node.height)
                slide_dict['depth'].append(node.depth)
                slide_dict['quality'] = 'Bad' if node.is_bad_quality else 'OK'
            slide_dict['id'] = list(
                i for i in range(len(slide_dict['x_coord'])))
        else:
            thumb_level = 1
            scale_factor = self.image.level_dimensions[0][0] // self.image.level_dimensions[thumb_level][0]
            thumb = np.array(self.image.get_thumbnail(
                self.image.level_dimensions[thumb_level]))

            for root in self.roots:
                queue = [root]
            while queue:
                node = queue.pop(0)
                queue.extend(node.get_children())
                if type_function is not None:
                    tumor_type, area = type_function(
                        node.x, node.y, node.width, node.height)
                    node.set_type(tumor_type, area)
                    slide_dict['type'].append(tumor_type)
                    slide_dict['damaged_amount'].append(area)
                else:
                    slide_dict['type'].append(np.nan)
                    slide_dict['damaged_amount'].append(np.nan)

                slide_dict['x_coord'].append(node.x)
                slide_dict['y_coord'].append(node.y)
                slide_dict['width'].append(node.width)
                slide_dict['height'].append(node.height)
                slide_dict['depth'].append(node.depth)

                x = node.x // scale_factor
                y = node.y // scale_factor
                w = node.width // scale_factor
                h = node.height // scale_factor

                if quality_func is not None:
                    quality = quality_func(thumb[y:y+h, x:x+w])
                    if quality == 'Bad':
                        node.set_bad_quality()
                    else:
                        quality = 'OK'
                else:
                    quality = 'OK'
                slide_dict['quality'].append(quality)
            slide_dict['id'] = list(
                i for i in range(len(slide_dict['x_coord'])))
        return slide_dict

    def apply_analysis_to_quads(self, analyzer, zoom_levels, batch_size):
        """Applies an analysis to all quads within the WSI.

        Args:
            analyzer (Analyzer): the analyzer to apply the predictions to
            zoom_levels (list): zoom levels to use
            batch_size (int): batch size to use when applying the analysis
        """
        self.analyses_classes_ordering[analyzer.name] = analyzer.classes
        num_roots = len(self.roots)
        input_shape = (analyzer.input_shape[0], analyzer.input_shape[1])
        for j, root in enumerate(self.roots):
            analyzed_region = self.image.read_region(
                (root.x, root.y), 0, (root.width, root.height))
            color = (255, 255, 255)
            image = Image.new('RGB', analyzed_region.size, color)
            image.paste(analyzed_region, mask=analyzed_region.split()[3])
            image = np.array(image) / 255.

            queue = [root]
            nodes = []
            while queue:
                node = queue.pop(0)
                node_children = node.get_children()
                queue.extend(node_children)
                if node.depth in zoom_levels:
                    nodes.append(node)

            sys.stdout.flush()
            for i in range((len(nodes) // batch_size) + 1):
                nds = nodes[i*batch_size:(i+1)*batch_size]
                sys.stdout.write(
                    "\r Analyzing regions, root {} / {}, nodes {} - {}".format(j + 1, num_roots, i*batch_size, (i+1)*batch_size))
                batch = []
                for n in nds:
                    x = n.x - root.x
                    y = n.y - root.y
                    node_image = image[y:y+n.height, x:x+n.width]
                    node_image_resized = cv2.resize(
                        node_image, dsize=input_shape, interpolation=cv2.INTER_AREA)
                    batch.append(node_image_resized)
                predictions = analyzer.net.predict(
                    np.array(batch), batch_size=batch_size)
                for node, pred in zip(nds, predictions):
                    node.set_analysis_for_analyzer(analyzer.name, pred)
