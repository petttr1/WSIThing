from openslide.deepzoom import DeepZoomGenerator
import openslide as ops
from lib.Utils.utils import slugify
from lib.WSITools.wsi_analyzer import WSIAnalyzer


class FeWSIMaker:

    analyzer = None

    def __init__(self, path_to_slide, slide_name, annotation_parser=None):
        self.path_to_slide = path_to_slide
        if annotation_parser is not None:
            self.annotation_parser = annotation_parser
        self.wsi = self.__load_slide()
        self.slide_name = slide_name.split('.')[0].lower()
        self.config = {
            'tile_size': 240,
            'overlap': 1,
            'limit_bounds': True
        }
        self.associated_images = []
        self.slides = {self.slide_name: DeepZoomGenerator(
            self.wsi, **self.config)}
        for name, image in self.wsi.associated_images.items():
            self.associated_images.append(name)
            slug = slugify(name)
            self.slides[slug] = DeepZoomGenerator(
                ops.ImageSlide(image), **self.config)
        try:
            mpp_x = self.wsi.properties[ops.PROPERTY_NAME_MPP_X]
            mpp_y = self.wsi.properties[ops.PROPERTY_NAME_MPP_Y]
            self.slide_mpp = (float(mpp_x) + float(mpp_y)) / 2
        except (KeyError, ValueError):
            self.slide_mpp = 0

    def make_analyzer(self, dump):
        try:
            self.analyzer = WSIAnalyzer(
                self.path_to_slide, annotation_parser=self.annotation_parser, quad_bunch_json_export=dump, wsi=self.wsi)
            return {'status': 'success', 'analyses': self.analyzer.available_analyses}
        except Exception as e:
            raise
            return {'status': 'fail', 'message': str(e)}

    def get_analysis(self, name, threshold):
        return self.analyzer.get_analysis_for_expert(name, thresh=threshold)

    def get_classes(self, analyzer):
        return self.analyzer.wsi_quad.analyses_classes_ordering[analyzer]

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

    def get_size(self):
        return self.wsi.level_dimensions[0]
