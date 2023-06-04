import numpy as np
import pandas as pd
from PIL import Image
from random import shuffle


class RootBatchProducer:
    def __init__(self, df, source, analyzer_name, gt_name='gt', generate_normal=False, roots=[]):
        self.df = df
        self.source = source
        self.analyzer_name = analyzer_name
        self.gt_name = gt_name
        self.generate_normal = generate_normal
        self.num_roots = df.index[-1][0]
        if len(roots) > 0:
            self.roots_to_use = roots
        else:
            self.roots_to_use = self.__filter_empty_roots()
        shuffle(self.roots_to_use)
        self.current_root = -1

    def __filter_empty_roots(self):
        if self.generate_normal == False:
            to_use = []
            for root in range(self.num_roots):
                root_loc = self.df.loc[pd.IndexSlice[root, :], :]
                leaf_loc = root_loc[[
                    np.argmax(d[self.gt_name]) != 3 for d in root_loc.analyses]]
                if len(leaf_loc) > 0:
                    to_use.append(root)
            return to_use
        else:
            return list(range(self.num_roots))

    def __prepare_image(self, window):
        # 4 to 3 channels hack
        color = (255, 255, 255)
        image = Image.new('RGB', window.size, color)
        # 3 is the alpha channel
        image.paste(window, mask=window.split()[3])
        return np.array(image)

    def read_window(self, source, x, y, w, h):
        window = source.read_region((x, y), 0, (w, h))
        return self.__prepare_image(window)

    def get_next_root(self):
        self.current_root += 1
        self.current_root %= len(self.roots_to_use)
        return self.roots_to_use[self.current_root]

    def get_window(self, node, root, region, size):
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

    def generate_batch(self, root, batch_size, size, num_classes):
        def cache_image(image, idx, cache):
            if idx not in cache.keys():
                cache[idx] = image

        def try_get_cached(idx, cahce):
            try:
                return cache[idx]
            except:
                return None

        # Create the dict for caching images
        cache = {}
        # Select all images for root *root*
        root_loc = self.df.loc[pd.IndexSlice[root, :], :]
        if self.generate_normal == False:
            leaf_loc = root_loc[[
                np.argmax(d[self.gt_name]) != 3 for d in root_loc.analyses]]
        else:
            leaf_loc = root_loc
        # Sample *batch_size* leaves
        leaves = leaf_loc[leaf_loc.is_leaf == True].sample(
            batch_size, replace=True)
        batch = []
        analyses = []
        gts = []
        # Select the root
        r = root_loc.iloc[0]
        # Read the region from which all the images are selected
        im = self.read_window(self.source, r.x, r.y, r.width, r.height)
        # for each samples leaf:
        for _, row in leaves.iterrows():
            ims = []
            an = []
            # read the leaf image
            ims.append(self.get_window(row, r, im, size))
            n_classes = num_classes
            gt = np.array(row.analyses[self.gt_name]
                          [:n_classes], dtype=np.float32)
            gt = np.repeat(gt.reshape(1, -1), 6, 0)
            an.append(row.analyses[self.analyzer_name])
            # read all of its parents
            for parent in row.parents[::-1]:
                p = root_loc.loc[pd.IndexSlice[root, parent], :]
                # Try to get cahced version first, if failed, load the image and cache it
                cached = try_get_cached(parent, cache)
                if cached is None:
                    window = self.get_window(p, r, im, size)
                    ims.append(window)
                    cache_image(window, parent, cache)
                else:
                    ims.append(cached)
                an.append(p.analyses[self.analyzer_name])
            # Append the slice to batch
            batch.append(ims)
            gts.append(gt)
            analyses.append(np.array(an))
        np_batch = np.array(batch)
        np_batch = np.swapaxes(np_batch, 0, 1)
        # np_batch = np.concatenate(np_batch, axis=3)
        return [b for b in np_batch], np.array(list(zip(analyses, gts)))


class WeightGenerator:
    def __init__(self, analyzers, batch_size, size, analyzer_name, num_classes, select_roots=[]):
        self.select_roots = select_roots
        self.num_classes = num_classes
        if num_classes == 4:
            self.generate_normal = True
        else:
            self.generate_normal = False
        self.analyzers = analyzers
        self.analyzer_name = analyzer_name
        self.batch_size = batch_size
        self.size = size
        self.batch_producers = []
        self.__create_batch_producers()
        self.current_producer = 0

    def generate(self, eval_data=False):
        while True:
            producer = self.batch_producers[self.current_producer]
            next_root = producer.get_next_root()
            if eval_data == True:
                # if eval, generate double batch and
                yield producer.generate_batch(next_root, self.batch_size, self.size, self.num_classes)
                # remove the root generated from generating pool
                producer.roots_to_use.pop(producer.current_root)
            else:
                yield producer.generate_batch(next_root, self.batch_size, self.size, self.num_classes)
            self.current_producer += 1
            self.current_producer %= len(self.batch_producers)

    def __create_batch_producers(self):
        if len(self.select_roots) != len(self.analyzers):
            for analyzer in self.analyzers:
                xport = analyzer.wsi_quad.export_json()
                df = self.extract_from_multiple_roots(xport['roots'])
                batch_producer = RootBatchProducer(
                    df, analyzer.wsi, self.analyzer_name, generate_normal=self.generate_normal)
                self.batch_producers.append(batch_producer)
        else:
            for analyzer, roots in zip(self.analyzers, self.select_roots):
                xport = analyzer.wsi_quad.export_json()
                df = self.extract_from_multiple_roots(xport['roots'])
                batch_producer = RootBatchProducer(
                    df, analyzer.wsi, self.analyzer_name, roots=roots, generate_normal=self.generate_normal)
                self.batch_producers.append(batch_producer)

    def extract_from_multiple_roots(self, roots):
        schema = {
            'root': [],
            'idx': [],
            'x': [],
            'y': [],
            'height': [],
            'width': [],
            'granularity': [],
            'parents': [],
            'depth': [],
            'bad_quality': [],
            'analyses': [],
            'is_leaf': []
        }
        for root in roots:
            schema = self.extract_to_df(root, schema)
        return pd.DataFrame(schema).set_index(['root', 'idx'])

    def extract_to_df(self, json, schema):
        schema['root'].append(json['root'])
        schema['idx'].append(json['idx'])
        schema['x'].append(json['x'])
        schema['y'].append(json['y'])
        schema['height'].append(json['height'])
        schema['width'].append(json['width'])
        schema['granularity'].append(json['granularity'])
        schema['parents'].append(json['parents'])
        schema['depth'].append(json['depth'])
        schema['bad_quality'].append(json['bad_quality'])
        schema['analyses'].append(json['analyses'])
        if json['depth'] == 5:
            schema['is_leaf'].append(True)
        else:
            schema['is_leaf'].append(False)
            for c in json['children']:
                self.extract_to_df(c, schema)
        return schema
