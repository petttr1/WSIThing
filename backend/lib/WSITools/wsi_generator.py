import numpy as np
import math
from keras.utils import to_categorical


class WSIGenerator:
    classes = None

    def __init__(self, analyzers, batch_size=64, output_size=256, preprocess_function=None, zoom_levels=[], tumor_types='auto', use_low_quality=True):
        """Generator of patches from WSIs. Uses analyzers to get images of different WSIs and combines them for variations.

        Args:
            analyzers (list): analyzers which are used to create a generator
            batch_size (int, optional): size of generated batches. Defaults to 64.
            output_size (int, optional): Size of one side of the image generated by the generator. Defaults to 256.
            preprocess_function (function, optional): function applied to each generated image. Defaults to None.
            zoom_levels (list, optional): zoom levels to use. Defaults to [].
            tumor_types (str, optional): 'auto' or dict of tumor type name and corresponding list of tumor types from the WSI Analyzers. Defaults to 'auto'.
            use_low_quality (bool, optional): Whether to generate low quality images. Defaults to True.
        """

        self.num_classes = None
        self.analyzers = analyzers
        self.batch_size = batch_size
        self.preprocess_function = preprocess_function
        self.zoom_levels = zoom_levels
        self.tumor_types = tumor_types
        self.output_size = output_size
        self.quality_options = [
            'OK', 'Bad'] if use_low_quality == True else ['OK']
        self.__unify_classes()
        self.analyzers_registrations = {tumor_type: {'analyzers': [],
                                                     'next': 0
                                                     } for tumor_type in self.tumor_types}
        self.__register_analyzers()
        self.__prepare_batch_creators()
        self.categorical_class_mapping = {
            c: i for i, c in enumerate(self.tumor_types.keys())}

########## GENERATOR UTILS ##########

    def get_classes(self):
        return self.classes

    def set_classes(self, classes):
        """Sets the classes for the current generator.

        Args:
            classes (dict): [description]
        """
        self.classes = classes
        for analyzer in self.analyzers:
            analyzer.set_classes(self.classes)

    def __get_next_analyzer_class(self, cclass):
        """Cycles through analyzers to vary the images.

        Args:
            cclass (str): class for which to get a new analyzer

        Returns:
            Analyzer: a new analyzer to use to request images from
        """
        self.analyzers_registrations[cclass]['next'] += 1
        self.analyzers_registrations[cclass]['next'] %= len(
            self.analyzers_registrations[cclass]['analyzers'])
        return self.analyzers[self.analyzers_registrations[cclass]
                              ['analyzers'][self.analyzers_registrations[cclass]['next'] - 1]]

    def __unify_classes(self):
        """Unifies classes from different WSIs. Needed for properly working *to_categorical* function.
        """
        if isinstance(self.tumor_types, dict):
            self.classes = {}
            for k, v in self.tumor_types.items():
                for ttype in v:
                    self.classes[ttype] = k
        else:
            tumor_types = []
            for analyzer in self.analyzers:
                tumor_types.extend(analyzer.cancer_types)
            # creates dict {type: number} with entry for each tumor type
            self.classes = {tumor_type: one_hot for one_hot,
                            tumor_type in enumerate(list(set(tumor_types)))}
        self.num_classes = len(list(self.classes.values()))
        for analyzer in self.analyzers:
            analyzer.set_classes(self.classes)

    def __register_analyzers(self):
        """Registers analyzers for usage in this generator.
        """
        for i, analyzer in enumerate(self.analyzers):
            temp_classes = analyzer.cancer_types
            for tumor_type, cclass in list(self.classes.items()):
                if tumor_type in temp_classes and i not in self.analyzers_registrations[cclass]['analyzers']:
                    self.analyzers_registrations[cclass]['analyzers'].append(
                        i)

    def __prepare_batch_creators(self):
        """Prepares the batch creators which in turn together produce the batches of images + labels. 
        """
        def tumor_type_generator():
            while True:
                for tumor_type in list(self.classes.keys()):
                    yield tumor_type
        self.type_gen = tumor_type_generator()
        if self.tumor_types == 'auto':
            self.tumor_types = {c_type: [c_type]
                                for c_type in list(self.classes.keys())}
        for analyzer in self.analyzers:
            analyzer.register_batch_producers(
                self.tumor_types, self.zoom_levels, self.quality_options, math.ceil(self.batch_size / len(self.tumor_types.keys())))

########## DATA GENERATION ##########
    def generate(self, eval_data=False):
        """Endless generator of data.
        """
        while True:
            yield self.__get_batch(eval_data=eval_data)

    def generate_eval_batch(self, num_batches):
        """Generates the Eval data

        Args:
            num_batches (int): number of batches to generate. The batch size is the same the generator was created with.

        Returns:
            tuple: tuple of images and their labels
        """
        imgs = np.zeros((num_batches*self.batch_size,
                         256, 256, 3), dtype=np.float64)
        labels = []
        for i, (im, lab) in enumerate(self.generate(eval_data=True)):
            if i >= num_batches:
                break
            imgs[i*self.batch_size:i*self.batch_size+self.batch_size] = im
            labels.extend(lab)
        return imgs, np.array(labels)

########## MULTICLASS CLASSIFIER GENERATION ##########

    def __get_batch(self, eval_data=False):
        """Prepares and returns a batch of images.

        Args:
            eval_data (bool, optional): Whether to use the generated images as eval data. If yes, then the images are excluded from generating again. Defaults to False.

        Returns:
            np.array: batch of images and labels
        """
        images, labels = [], []
        # For each tumor type registered:
        for tumor_type in list(set(self.classes.values())):
            # gets batch_size / num_classes images and labels to add to the batch
            next_analyzer = self.__get_next_analyzer_class(tumor_type)
            imgs, lbls = next_analyzer.prepare_samples(
                tumor_type, self.output_size, eval_data=eval_data)
            images.extend(imgs)
            labels.extend(lbls)
        if self.preprocess_function != None:
            # if preprocessing function is defines, it is applied
            preprocessed_batch = self.preprocess_function(
                np.array(images[:self.batch_size]))
        else:
            preprocessed_batch = np.array(images[:self.batch_size])

        # batch of images and labels is returned
        labels = to_categorical(
            list(map(lambda x: self.categorical_class_mapping[x], labels[:self.batch_size])))
        return preprocessed_batch, labels