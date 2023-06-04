from skimage.exposure import histogram
import numpy as np
from unicodedata import normalize
import re


def eval_window_quality(window):
    """Evaluates the window quality based on the ratio of white / non-white pixels using a histogram"""
    hist = histogram(window)
    if sum(np.where(hist[1] > 200, hist[0], 0)) > sum(np.where(hist[1] <= 200, hist[0], 0))*5:
        return 'Bad'
    return 'OK'


def slugify(text):
    text = normalize('NFKD', text.lower()).encode(
        'ascii', 'ignore').decode()
    return re.sub('[^a-z0-9]+', '-', text)
