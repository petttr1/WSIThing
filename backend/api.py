import os
import pickle
from io import BytesIO
from flask import Flask, request, url_for, send_from_directory, abort
from flask.helpers import make_response
from werkzeug.utils import secure_filename
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from lib.lib import create_storage, create_fe_analyzer, create_annotation
from lib.Utils.utils import slugify

UPLOAD_FOLDER = 'uploaded_files'
DEMO_FOLDER = 'demo_slides'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'svs'])
ALLOWED_EXTENSIONS_ANNOT = set(['xml', 'json'])
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config["DEBUG"] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DEMO_FOLDER'] = DEMO_FOLDER
CORS(app, supports_credentials=True)
socketio = SocketIO(
    app, cors_allowed_origins="*", engineio_logger=True)
# cors_allowed_origins="http://192.168.0.105:8080"
STORAGE = create_storage()


@app.route('/')
def index():
    return {'message': 'Hello World!'}


@app.route('/<slug>.dzi')
def dzi(slug):
    # TODO: hard-coded JPEG
    format = 'jpeg'
    try:
        resp = make_response(STORAGE.get(slug.split(
            '-')[0]).slides[slug].get_dzi(format))
        resp.mimetype = 'application/xml'
        return resp
    except KeyError:
        # Unknown slug
        abort(404)


@app.route('/<slug>_files/<int:level>/<int:col>_<int:row>.<format>')
def tile(slug, level, col, row, format):
    format = format.lower()
    if format != 'jpeg' and format != 'png':
        # Not supported by Deep Zoom
        abort(404)
    try:
        tile = STORAGE.get(slug.split(
            '-')[0]).slides[slug.split('-')[0]].get_tile(level, (col, row))
    except KeyError:
        # Unknown slug
        abort(404)
    except ValueError:
        # Invalid level or coordinates
        abort(404)
    buf = BytesIO()
    # TODO: hard-coded 75
    tile.save(buf, format, quality=75)
    resp = make_response(buf.getvalue())
    resp.mimetype = 'image/%s' % format
    return resp


@app.route('/<slug>_annot.png')
def simple_image(slug):
    try:
        image = STORAGE.get(slug + '_annot')
    except KeyError:
        # Unknown slug
        abort(404)
    resp = make_response(image.getvalue())
    resp.mimetype = 'image/png'
    return resp


@app.route('/upload_wsi', methods=['POST'])
def upload_file():
    print(' * received form with', list(request.form.items()))
    socketio.emit('statusMessage', 'File Received on the Backend')
    # check if the post request has the file part
    files = request.files.getlist('files')
    for wsi in list(filter(lambda x: x.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS, files)):
        # process the annotation belonging to the WSI
        annot_name = wsi.filename.split('.')[0].lower()
        annot_file = list(filter(lambda x: x.filename.split(
            '.')[-1].lower() in ALLOWED_EXTENSIONS_ANNOT and x.filename.split('.')[0].lower() == annot_name, files))[0]
        annot_filename = secure_filename(annot_file.filename).lower()
        annot_file.save(os.path.join(
            app.config['UPLOAD_FOLDER'], annot_filename))
        polygons, annot_parser, classes = create_annotation(
            os.path.join(app.config['UPLOAD_FOLDER'], annot_filename))
        socketio.emit('annotationProcessed', {'annot': {
            'objects': polygons,
            'classes': classes,
            'id': 'default'
        }})

        # process the actual WSI, using the annotation when creating the Analyzer
        filename = secure_filename(wsi.filename).lower()
        wsi.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print(' * file uploaded, creating FE WSI Analyzer', filename)
        socketio.emit('statusMessage', 'File Uploaded, Creating FE Analyzer')
        analyzer = create_fe_analyzer(os.path.join(
            app.config['UPLOAD_FOLDER'], filename), filename, annot_parser)
        storage_key = STORAGE.add_to_storage(
            analyzer, filename.split('.')[0])
        level_dimensions = STORAGE.get(storage_key).get_size()
        socketio.emit('wsiProcessed', {
            'wsi': {
                'Image': {
                    'xmlns': 'http://schemas.microsoft.com/deepzoom/2008',
                    'Url': 'http://127.0.0.1:5000/{}_files/'.format(slugify(filename)),
                    # TODO: hard-coded format
                    'Format': 'jpeg',
                    'Overlap': STORAGE.get(storage_key).config['overlap'],
                    'TileSize': STORAGE.get(storage_key).config['tile_size'],
                    'filename': filename,
                    'storageKey': storage_key,
                    'Size': {
                        'Width': level_dimensions[0],
                        'Height': level_dimensions[1]
                    }
                }
            }})
        dump_name = wsi.filename.split('.')[0].lower()
        dump_file = list(filter(lambda x: x.filename.split(
            '.')[-1].lower() == 'pickle' and x.filename.split('.')[0].lower() == dump_name, files))[0]
        dump_filename = secure_filename(dump_file.filename).lower()
        dump_file.save(os.path.join(
            app.config['UPLOAD_FOLDER'], dump_filename))
        print(' * dump file uploaded, creating WSI Analyzer', dump_filename)
        socketio.emit('statusMessage', 'Creating an Analyzer {}'.format(
            os.path.join(app.config['UPLOAD_FOLDER'], dump_filename)))
        # TODO: Hack af
        with open(os.path.join(app.config['UPLOAD_FOLDER'], dump_filename), 'rb') as handle:
            b = pickle.load(handle)
        result = STORAGE.get(storage_key).make_analyzer(b)
        if result['status'] == 'success':
            socketio.emit('analyzerCreated', result['analyses'])
        else:
            socketio.emit('statusMessage', 'Failed to create the analyzer.')
        # TODO: processes just one file for now
        break

    return {
        'result': 'success'
    }


@app.route('/get_demo_slides', methods=['GET'])
def get_all_demo_slide_names():
    files = list(map(lambda x: x.split('.')[0], filter(lambda x: x.split(
        '.')[-1] == 'svs', os.listdir(app.config['DEMO_FOLDER']))))
    return {'slides': files}


@app.route('/demo_slide/<string:name>', methods=['GET'])
def load_demo_slide(name):
    print(' * loading demo slide', name)
    socketio.emit('statusMessage', 'Request on the Backend')
    # TODO: hard-coded xml annot (probably ok?)
    polygons, annot_parser, classes = create_annotation(
        os.path.join(app.config['DEMO_FOLDER'], name + '.xml'))
    socketio.emit('annotationProcessed', {
        'annot': {
            'objects': polygons,
            'classes': classes,
            'id': 'default'
        }
    })
    # process the actual WSI, using the annotation when creating the Analyzer
    socketio.emit('statusMessage', 'File Registered, Creating FE Analyzer')
    # TODO: Hard-coded WSI format
    filename = name + '.svs'
    analyzer = create_fe_analyzer(os.path.join(
        app.config['DEMO_FOLDER'], filename), filename, annot_parser)
    storage_key = STORAGE.add_to_storage(analyzer, name)
    level_dimensions = STORAGE.get(storage_key).get_size()
    socketio.emit('wsiProcessed', {
        'wsi': {
            'Image': {
                'xmlns': 'http://schemas.microsoft.com/deepzoom/2008',
                'Url': 'http://127.0.0.1:5000/{}_files/'.format(slugify(filename)),
                # TODO: hard-coded format
                'Format': 'jpeg',
                'Overlap': STORAGE.get(storage_key).config['overlap'],
                'TileSize': STORAGE.get(storage_key).config['tile_size'],
                'filename': name + '.svs',
                'storageKey': storage_key,
                'Size': {
                    'Width': level_dimensions[0],
                    'Height': level_dimensions[1]
                }
            }
        }})
    dump_filename = name + '.pickle'
    socketio.emit('statusMessage', 'Creating an Analyzer {}'.format(
        os.path.join(app.config['DEMO_FOLDER'], dump_filename)))
    # TODO: Hack af // maybe not
    with open(os.path.join(app.config['DEMO_FOLDER'], dump_filename), 'rb') as handle:
        b = pickle.load(handle)
        result = STORAGE.get(storage_key).make_analyzer(b)
    if result['status'] == 'success':
        socketio.emit('analyzerCreated', result['analyses'])
    else:
        socketio.emit('statusMessage', 'Failed to create the analyzer.')

    return {
        'result': 'success'
    }


@app.route('/create_analyzer/<string:storage_key>', methods=['POST'])
def create_analyzer(storage_key):
    files = request.files.getlist('files')
    for dump in files:
        if dump.filename.split('.')[-1].lower() == 'pickle':
            filename = secure_filename(dump.filename).lower()
            dump.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(' * file uploaded, creating WSI Analyzer', filename)
            # TODO: Hack af
            with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as handle:
                b = pickle.load(handle)
            result = STORAGE.get(storage_key).make_analyzer(b)
        break
    return result


@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


@app.route('/get_analysis/<string:storage_key>/<string:analysis_name>', methods=['GET'])
def get_analysis(storage_key, analysis_name):
    threshold = request.args.get('threshold', default=0.5, type=float)
    try:
        overlay = STORAGE.get(storage_key).get_analysis(
            analysis_name, threshold)
        classes = STORAGE.get(storage_key).get_classes(analysis_name)
        socketio.emit('newOverlayLoaded', {
            'overlay': {
                'objects': overlay,
                'classes': classes,
                'id': analysis_name
            }
        })
        return {'result': 'success'}
    except:
        raise
        return {'result': 'fail'}


@socketio.on('connect')
def frontend_connect():
    print('FE Connected')
    emit('statusMessage', 'BE Connected, Please Upload a WSI')


if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000)
