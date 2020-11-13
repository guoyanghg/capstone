import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
import tfsource.architectures as architectures
from tfsource.utils import my_func, resolve_status, contoured_image

#### resnet pytorch import ####
import torch
import cv2 as cv
from torchsource.dataset import mask_convert, visualize
from torchsource.model import resnet18
import numpy as np


### torch ###
model = resnet18()
best_model_path = '../network/pytorch_resnet/bestmodel_255_resnet.pth'
checkpoint = torch.load(best_model_path)
# initialize state_dict from checkpoint to model
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()
### torch ###

### tensorflow ###
TRAINING_STATUS = 3  ### inference
restore, is_training = resolve_status(TRAINING_STATUS)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
fast_lookup = True
iter_limit = 300


def re_init_phi(phi, dt):
    D_left_shift = tf.cast(tf.manip.roll(phi, -1, axis=1), dtype='float32')
    D_right_shift = tf.cast(tf.manip.roll(phi, 1, axis=1), dtype='float32')
    D_up_shift = tf.cast(tf.manip.roll(phi, -1, axis=0), dtype='float32')
    D_down_shift = tf.cast(tf.manip.roll(phi, 1, axis=0), dtype='float32')
    bp = D_left_shift - phi
    cp = phi - D_down_shift
    dp = D_up_shift - phi
    ap = phi - D_right_shift
    an = tf.identity(ap)
    bn = tf.identity(bp)
    cn = tf.identity(cp)
    dn = tf.identity(dp)
    ap = tf.clip_by_value(ap, 0, 10 ^ 38)
    bp = tf.clip_by_value(bp, 0, 10 ^ 38)
    cp = tf.clip_by_value(cp, 0, 10 ^ 38)
    dp = tf.clip_by_value(dp, 0, 10 ^ 38)
    an = tf.clip_by_value(an, -10 ^ 38, 0)
    bn = tf.clip_by_value(bn, -10 ^ 38, 0)
    cn = tf.clip_by_value(cn, -10 ^ 38, 0)
    dn = tf.clip_by_value(dn, -10 ^ 38, 0)
    area_pos = tf.where(phi > 0)
    area_neg = tf.where(phi < 0)
    pos_y = area_pos[:, 0]
    pos_x = area_pos[:, 1]
    neg_y = area_neg[:, 0]
    neg_x = area_neg[:, 1]
    tmp1 = tf.reduce_max([tf.square(tf.gather_nd(t, area_pos)) for t in [ap, bn]], axis=0)
    tmp1 += tf.reduce_max([tf.square(tf.gather_nd(t, area_pos)) for t in [cp, dn]], axis=0)
    update1 = tf.sqrt(tf.abs(tmp1)) - 1
    indices1 = tf.stack([pos_y, pos_x], 1)
    tmp2 = tf.reduce_max([tf.square(tf.gather_nd(t, area_neg)) for t in [an, bp]], axis=0)
    tmp2 += tf.reduce_max([tf.square(tf.gather_nd(t, area_neg)) for t in [cn, dp]], axis=0)
    update2 = tf.sqrt(tf.abs(tmp2)) - 1
    indices2 = tf.stack([neg_y, neg_x], 1)
    indices_final = tf.concat([indices1, indices2], 0)
    update_final = tf.concat([update1, update2], 0)
    dD = tf.scatter_nd(indices_final, update_final, shape=[input_image_size, input_image_size])
    S = tf.divide(phi, tf.square(phi) + 1)
    phi = phi - tf.multiply(dt * S, dD)

    return phi


def get_curvature(phi, x, y):
    phi_shape = tf.shape(phi)
    dim_y = phi_shape[0]
    dim_x = phi_shape[1]
    x = tf.cast(x, dtype="int32")
    y = tf.cast(y, dtype="int32")
    y_plus = tf.cast(y + 1, dtype="int32")
    y_minus = tf.cast(y - 1, dtype="int32")
    x_plus = tf.cast(x + 1, dtype="int32")
    x_minus = tf.cast(x - 1, dtype="int32")
    y_plus = tf.minimum(tf.cast(y_plus, dtype="int32"), tf.cast(dim_y - 1, dtype="int32"))
    x_plus = tf.minimum(tf.cast(x_plus, dtype="int32"), tf.cast(dim_x - 1, dtype="int32"))
    y_minus = tf.maximum(y_minus, 0)
    x_minus = tf.maximum(x_minus, 0)
    d_phi_dx = tf.gather_nd(phi, tf.stack([y, x_plus], 1)) - tf.gather_nd(phi, tf.stack([y, x_minus], 1))
    d_phi_dx_2 = tf.square(d_phi_dx)
    d_phi_dy = tf.gather_nd(phi, tf.stack([y_plus, x], 1)) - tf.gather_nd(phi, tf.stack([y_minus, x], 1))
    d_phi_dy_2 = tf.square(d_phi_dy)
    d_phi_dxx = tf.gather_nd(phi, tf.stack([y, x_plus], 1)) + tf.gather_nd(phi, tf.stack([y, x_minus], 1)) - \
                2 * tf.gather_nd(phi, tf.stack([y, x], 1))
    d_phi_dyy = tf.gather_nd(phi, tf.stack([y_plus, x], 1)) + tf.gather_nd(phi, tf.stack([y_minus, x], 1)) - \
                2 * tf.gather_nd(phi, tf.stack([y, x], 1))
    d_phi_dxy = 0.25 * (- tf.gather_nd(phi, tf.stack([y_minus, x_minus], 1)) - tf.gather_nd(phi, tf.stack(
        [y_plus, x_plus], 1)) + tf.gather_nd(phi, tf.stack([y_minus, x_plus], 1)) + tf.gather_nd(phi, tf.stack(
        [y_plus, x_minus], 1)))
    tmp_1 = tf.multiply(d_phi_dx_2, d_phi_dyy) + tf.multiply(d_phi_dy_2, d_phi_dxx) - \
            2 * tf.multiply(tf.multiply(d_phi_dx, d_phi_dy), d_phi_dxy)
    tmp_2 = tf.add(tf.pow(d_phi_dx_2 + d_phi_dy_2, 1.5), 2.220446049250313e-16)
    tmp_3 = tf.pow(d_phi_dx_2 + d_phi_dy_2, 0.5)
    tmp_4 = tf.divide(tmp_1, tmp_2)
    curvature = tf.multiply(tmp_3, tmp_4)
    mean_grad = tf.pow(d_phi_dx_2 + d_phi_dy_2, 0.5)

    return curvature, mean_grad


def get_intensity(image, masked_phi, filter_patch_size=5):
    u_1 = tf.layers.average_pooling2d(tf.multiply(image, masked_phi), [filter_patch_size, filter_patch_size], 1,
                                      padding='SAME')
    u_2 = tf.layers.average_pooling2d(masked_phi, [filter_patch_size, filter_patch_size], 1, padding='SAME')
    u_2_prime = 1 - tf.cast((u_2 > 0), dtype='float32') + tf.cast((u_2 < 0), dtype='float32')
    u_2 = u_2 + u_2_prime + 2.220446049250313e-16

    return tf.divide(u_1, u_2)


def active_contour_layer(elems):
    img = elems[0]
    init_phi = elems[1]
    map_lambda1_acl = elems[2]
    map_lambda2_acl = elems[3]
    wind_coef = 3
    zero_tensor = tf.constant(0, shape=[], dtype="int32")

    def _body(i, phi_level):
        band_index = tf.reduce_all([phi_level <= narrow_band_width, phi_level >= -narrow_band_width], axis=0)
        band = tf.where(band_index)
        band_y = band[:, 0]
        band_x = band[:, 1]
        shape_y = tf.shape(band_y)
        num_band_pixel = shape_y[0]
        window_radii_x = tf.ones(num_band_pixel) * wind_coef
        window_radii_y = tf.ones(num_band_pixel) * wind_coef

        def body_intensity(j, mean_intensities_outer, mean_intensities_inner):
            xnew = tf.cast(band_x[j], dtype="float32")
            ynew = tf.cast(band_y[j], dtype="float32")
            window_radius_x = tf.cast(window_radii_x[j], dtype="float32")
            window_radius_y = tf.cast(window_radii_y[j], dtype="float32")
            local_window_x_min = tf.cast(tf.floor(xnew - window_radius_x), dtype="int32")
            local_window_x_max = tf.cast(tf.floor(xnew + window_radius_x), dtype="int32")
            local_window_y_min = tf.cast(tf.floor(ynew - window_radius_y), dtype="int32")
            local_window_y_max = tf.cast(tf.floor(ynew + window_radius_y), dtype="int32")
            local_window_x_min = tf.maximum(zero_tensor, local_window_x_min)
            local_window_y_min = tf.maximum(zero_tensor, local_window_y_min)
            local_window_x_max = tf.minimum(tf.cast(input_image_size - 1, dtype="int32"), local_window_x_max)
            local_window_y_max = tf.minimum(tf.cast(input_image_size - 1, dtype="int32"), local_window_y_max)
            local_image = img[local_window_y_min: local_window_y_max + 1, local_window_x_min: local_window_x_max + 1]
            local_phi = phi_prime[local_window_y_min: local_window_y_max + 1,
                        local_window_x_min: local_window_x_max + 1]
            inner = tf.where(local_phi <= 0)
            area_inner = tf.cast(tf.shape(inner)[0], dtype='float32')
            outer = tf.where(local_phi > 0)
            area_outer = tf.cast(tf.shape(outer)[0], dtype='float32')
            image_loc_inner = tf.gather_nd(local_image, inner)
            image_loc_outer = tf.gather_nd(local_image, outer)
            mean_intensity_inner = tf.cast(tf.divide(tf.reduce_sum(image_loc_inner), area_inner), dtype='float32')
            mean_intensity_outer = tf.cast(tf.divide(tf.reduce_sum(image_loc_outer), area_outer), dtype='float32')
            mean_intensities_inner = tf.concat(axis=0, values=[mean_intensities_inner[:j], [mean_intensity_inner]])
            mean_intensities_outer = tf.concat(axis=0, values=[mean_intensities_outer[:j], [mean_intensity_outer]])

            return (j + 1, mean_intensities_outer, mean_intensities_inner)

        if fast_lookup:
            phi_4d = phi_level[tf.newaxis, :, :, tf.newaxis]
            image = img[tf.newaxis, :, :, tf.newaxis]
            band_index_2 = tf.reduce_all([phi_4d <= narrow_band_width, phi_4d >= -narrow_band_width], axis=0)
            band_2 = tf.where(band_index_2)
            u_inner = get_intensity(image, tf.cast((([phi_4d <= 0])), dtype='float32')[0], filter_patch_size=f_size)
            u_outer = get_intensity(image, tf.cast((([phi_4d > 0])), dtype='float32')[0], filter_patch_size=f_size)
            mean_intensities_inner = tf.gather_nd(u_inner, band_2)
            mean_intensities_outer = tf.gather_nd(u_outer, band_2)

        else:
            mean_intensities_inner = tf.constant([0], dtype='float32')
            mean_intensities_outer = tf.constant([0], dtype='float32')
            j = tf.constant(0, dtype=tf.int32)
            _, mean_intensities_outer, mean_intensities_inner = tf.while_loop(
                lambda j, mean_intensities_outer, mean_intensities_inner:
                j < num_band_pixel, body_intensity, loop_vars=[j, mean_intensities_outer, mean_intensities_inner],
                shape_invariants=[j.get_shape(), tf.TensorShape([None]), tf.TensorShape([None])])

        lambda1 = tf.gather_nd(map_lambda1_acl, [band])
        lambda2 = tf.gather_nd(map_lambda2_acl, [band])
        curvature, mean_grad = get_curvature(phi_level, band_x, band_y)
        kappa = tf.multiply(curvature, mean_grad)
        term1 = tf.multiply(tf.cast(lambda1, dtype='float32'),
                            tf.square(tf.gather_nd(img, [band]) - mean_intensities_inner))
        term2 = tf.multiply(tf.cast(lambda2, dtype='float32'),
                            tf.square(tf.gather_nd(img, [band]) - mean_intensities_outer))
        force = -nu + term1 - term2
        force /= (tf.reduce_max(tf.abs(force)))
        d_phi_dt = tf.cast(force, dtype="float32") + tf.cast(mu * kappa, dtype="float32")
        dt = .45 / (tf.reduce_max(tf.abs(d_phi_dt)) + 2.220446049250313e-16)
        d_phi = dt * d_phi_dt
        update_narrow_band = d_phi
        phi_prime = phi_level + tf.scatter_nd([band], tf.cast(update_narrow_band, dtype='float32'),
                                              shape=[input_image_size, input_image_size])
        phi_prime = re_init_phi(phi_prime, 0.5)

        return (i + 1, phi_prime)

    i = tf.constant(0, dtype=tf.int32)
    phi = init_phi
    _, phi = tf.while_loop(lambda i, phi: i < iter_limit, _body, loop_vars=[i, phi])
    phi = tf.round(tf.cast((1 - tf.nn.sigmoid(phi)), dtype=tf.float32))

    return phi, init_phi, map_lambda1_acl, map_lambda2_acl

### gy use my dataset ###
INPUT_CHANNEL = 3
IMAGE_SIZE = 256
BATCH_SIZE = 1
### gy use my dataset ###
config = tf.ConfigProto(allow_soft_placement=True)
input_shape = [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, INPUT_CHANNEL]
input_shape_dt = [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE]

narrow_band_width = 1
mu = 0.2
nu = 5.0
f_size = 15
input_image_size = IMAGE_SIZE

global graph
graph = tf.get_default_graph()
sess = tf.Session(graph=graph, config=config)

with graph.as_default():
    x = tf.placeholder(shape=input_shape, dtype=tf.float32, name="x")
    # y = tf.placeholder(dtype=tf.float32, name="y")
    phase = tf.placeholder(tf.bool, name='phase')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    out_seg, map_lambda1, map_lambda2 = architectures.ddunet(x, is_training)
    y_out_dl = tf.round(out_seg)
    x_acm = x[:, :, :, 0]
    rounded_seg_acl = y_out_dl[:, :, :, 0]
    dt_trans = tf.py_func(my_func, [rounded_seg_acl], tf.float32)
    dt_trans.set_shape([BATCH_SIZE, input_image_size, input_image_size])
    phi_out, _, lambda1_tr, lambda2_tr = tf.map_fn(fn=active_contour_layer, elems=(
        x_acm, dt_trans, map_lambda1[:, :, :, 0], map_lambda2[:, :, :, 0]))

    rounded_seg = tf.round(out_seg)
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, tf.train.latest_checkpoint('../network/model.ckpt'))

### tensorflow graph setting end here###


def inference_torch(image):
    print("########### Inference torch############")
    with torch.no_grad():
        image_tensor = torch.from_numpy(image).float()
        image_tensor = image_tensor.permute(0,3,1,2)
        image_tensor = image_tensor.cuda()
        pred_mask = model.forward(image_tensor)
        mask_img = mask_convert(pred_mask)
        mask_img = cv.normalize(mask_img, None, 0, 1, cv.NORM_MINMAX)

        ### rescale to 255 for saving ###
        mask_img = np.round(mask_img)*255

        ### find contour image ###
        contour_img = contoured_image(mask_img, image[0, :, :, 0])
        return mask_img, contour_img

def inference(image):
    print("########### Inference tf############")
    seg_out_acm, seg_out = sess.run([phi_out, y_out_dl], {x: image, phase: False})
    seg_out_acm = seg_out_acm[0, :, :]
    seg_out = seg_out[0, :, :, 0]

    seg_out_coutour = contoured_image(seg_out, image[0, :, :, 0])
    seg_out_acm_coutour = contoured_image(seg_out_acm, image[0, :, :, 0])

    return seg_out_coutour, seg_out_acm_coutour

from flask import jsonify, Flask, render_template, flash, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from flask_login import LoginManager, login_required, login_user, UserMixin
from skimage import io, transform
import MySQLdb
from datetime import timedelta

### app settings ###
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_FOLDER = './static/source'
RESULT_FOLDER = './static/result'
app = Flask(__name__)
app.secret_key ='205322607'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.before_request
def make_session_permanent():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=5)

### Login manager ###
login_manager = LoginManager()
login_manager.remember_cookie_duration=timedelta(days=1)
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, id, username, password, email):
        super().__init__()
        self.id = id
        self.username = username
        self.password = password
        self.email = email
    def is_anonymous(self):
        return False
    def is_active(self):
        return True
    def is_authenticated(self):
        return True
    def get_id(self):
        return self.id

@login_manager.unauthorized_handler
def unauthorized_callback():
    return redirect('/login')

@login_manager.user_loader
def load_user(user_id):
    cursor = db.cursor()
    cursor.execute(f"""SELECT * From User WHERE id = {user_id}""")
    data = cursor.fetchone()
    if data is None:
        return None
    id = str(data[0])
    user = User(id, username=data[1], password=data[2], email=data[3])
    return user
### Login manager ###

### DB connection ###
db = MySQLdb.connect(host="localhost",user="root",
                  passwd="root",db="nucleus")

### extension validation ###
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

### save images as npy ###
def read_img_to_npy(img_dir):
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    image = io.imread(img_dir)[:, :, :3].astype('float32')
    image = transform.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
    image = np.asarray([image] * 1)
    return image

@app.route('/')
def root_index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)

@app.route('/upload', methods=['GET'])
@login_required
def on_upload():
    try:
        return render_template('upload.html')
    except Exception as e:
        return str(e)

@app.route('/upload', methods=['POST'])
def show_result():
    img_dir = ''
    if 'imginput' not in request.files:
        flash('No file part')
        return redirect('/upload')
    file = request.files['imginput']
    method = request.form.get('method')
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect('/upload')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        img_dir = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_dir)

    ### read file then convert ###
    image = read_img_to_npy(img_dir)
    print(image.shape)

    ### find inference function ###
    seg_out_1, seg_out_2 = None, None
    title1 = None
    title2 = None

    if method == 'resnet':
        title1 = "Resnet MASK Output"
        title2 = "Resnet Contour Output"
        seg_out_1, seg_out_2 = inference_torch(image)
    else:
        title1 = "UNET Contour Output"
        title2 = "ACM Contour Output"
        seg_out_1, seg_out_2 = inference(image)

    ### SAVE IMAGE ###
    result_dir =img_dir.replace(app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER']).split('.')[1]

    unet_img = '.' + result_dir+'_unet_result.jpg'
    acm_img = '.'+ result_dir + '_acm_result.jpg'
    print(unet_img)
    print(acm_img)
    cv.imwrite(unet_img, seg_out_1)
    cv.imwrite(acm_img, seg_out_2)

    return render_template('result.html',
                           unet_url = unet_img,
                           acm_url = acm_img,
                           title1 = title1,
                           title2 = title2,
                           method = method.upper())

@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def validate_login():
    email = request.form.get('email')
    password = request.form.get('password')
    cursor = db.cursor()
    cursor.execute(f"""SELECT * From User WHERE email = '{email}'""")
    data = cursor.fetchone()
    if data is None:
        return redirect('/login')
    user_password = data[2]
    if user_password == password:
        id = str(data[0])
        user = User(id, data[1], data[2], data[3])
        login_user(user)
        return redirect('/')
    return redirect('/login')


@app.route('/mobile/login', methods=['POST'])
def validate_mobile_login():
    ### get json data ###
    content = request.get_json()
    print(content)
    login_req = content['request']
    email = login_req['email']
    password = login_req['password']

    ### query data from db ###
    cursor = db.cursor()
    cursor.execute(f"""SELECT * From User WHERE email = '{email}'""")
    data = cursor.fetchone()
    if data is None:
        print("No user")
        return jsonify({'status':'NU'})
    user_password = data[2]
    if user_password == password:
        #id = str(data[0])
        #user = User(id, data[1], data[2], data[3])
        #login_user(user)
        print("success")
        return jsonify({'status':'SC'})
    print("wrong password")
    return jsonify({'status':'WP'})




@app.route('/signup', methods=['GET'])
def signup_page():
    return render_template('signup.html')

@app.route('/signup', methods=['POST'])
def validate_signup():
    email = request.form.get('email')
    password = request.form.get('password')
    username = request.form.get('username')
    cursor = db.cursor()

    cursor.execute(f"""SELECT * From User WHERE email = '{email}'""")
    data = cursor.fetchone()
    ### email has already exist ###
    if data is not None:
        print('lol')
        return redirect('/signup')

    try:
        cursor.execute(f"""INSERT INTO User (username, email, password) VALUES ('{username}', '{email}', '{password}')""")
        db.commit()
    except:
        print('rollback')
        db.rollback()
        return redirect('/signup')

    cursor.execute(f"""SELECT * From User WHERE email = '{email}'""")
    data = cursor.fetchone()
    ### read the id imformation and login the user###
    id = str(data[0])
    user = User(id, data[1], data[2], data[3])
    login_user(user)
    return redirect('/')


@app.route('/test')
def model_testing():
    cursor = db.cursor()
    cursor.execute("""SELECT * From User""")
    data = cursor.fetchone()
    return 'testing template'

if __name__ == '__main__':
    # app.run(host, port, debug, options)
    # default：host="127.0.0.1", port=5000, debug=False
    app.run(host='0.0.0.0', port=5000)
    #app.run()
