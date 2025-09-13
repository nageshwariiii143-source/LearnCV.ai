import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
import tempfile

st.set_page_config(page_title="Image Processing Toolkit", layout="wide")

# ----------------------- Utility functions -----------------------

def to_pil(img: np.ndarray):
    """Convert OpenCV BGR/Gray numpy array to PIL Image (RGB)."""
    if img is None:
        return None
    if len(img.shape) == 2:
        return Image.fromarray(img)
    # assume BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def read_image(file) -> (np.ndarray, dict):
    """Read uploaded file into OpenCV BGR array and return metadata."""
    image = Image.open(file).convert('RGB')
    arr = np.array(image)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    meta = {
        'format': getattr(image, 'format', 'PNG'),
        'mode': image.mode,
        'size': image.size,  # (W,H)
    }
    return bgr, meta


def image_info(img_bgr, meta=None):
    if img_bgr is None:
        return {}
    h, w = img_bgr.shape[:2]
    channels = 1 if len(img_bgr.shape) == 2 else img_bgr.shape[2]
    info = {
        'Width': w,
        'Height': h,
        'Channels': channels,
    }
    if meta:
        info['Format'] = meta.get('format', '')
        info['Mode'] = meta.get('mode', '')
        info['Size'] = f"{meta.get('size', (w,h))}"
    return info


def convert_color(img_bgr, mode):
    if img_bgr is None:
        return None
    if mode == 'RGB':
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if mode == 'HSV':
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if mode == 'YCbCr':
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    if mode == 'Grayscale':
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr


def rotate_image(img, angle, center=None, scale=1.0):
    (h, w) = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, M, (w, h))


def scale_image(img, fx, fy):
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)


def translate_image(img, tx, ty):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    (h, w) = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))


def affine_transform(img, src_pts, dst_pts):
    M = cv2.getAffineTransform(np.float32(src_pts), np.float32(dst_pts))
    (h, w) = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))


def perspective_transform(img, src_pts, dst_pts):
    M = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    (h, w) = img.shape[:2]
    return cv2.warpPerspective(img, M, (w, h))


def apply_filter(img, filter_name, ksize=3):
    if img is None:
        return None
    if filter_name == 'Gaussian':
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if filter_name == 'Median':
        return cv2.medianBlur(img, ksize)
    if filter_name == 'Mean':
        return cv2.blur(img, (ksize, ksize))
    if filter_name == 'Sobel':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        mag = cv2.magnitude(sx, sy)
        mag = np.uint8(np.clip(mag, 0, 255))
        return mag
    if filter_name == 'Laplacian':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = np.uint8(np.clip(np.abs(lap), 0, 255))
        return lap
    return img


def morphology(img, op, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if op == 'Erode':
        return cv2.erode(img, kernel, iterations=1)
    if op == 'Dilate':
        return cv2.dilate(img, kernel, iterations=1)
    if op == 'Open':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    if op == 'Close':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


def histogram_equalization(img):
    if len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return cv2.equalizeHist(img)


def contrast_stretching(img):
    # simple linear contrast stretch per channel
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        channel = img[:, :, c]
        minv = np.min(channel)
        maxv = np.max(channel)
        if maxv - minv == 0:
            out[:, :, c] = channel
        else:
            out[:, :, c] = ((channel - minv) * 255.0 / (maxv - minv)).astype(np.uint8)
    return out


def sharpen_image(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def canny_edge(img, low, high):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    return edges


def save_bytes(img, fmt='PNG', quality=95):
    pil = to_pil(img)
    buf = io.BytesIO()
    save_kwargs = {}
    if fmt.upper() == 'JPEG' or fmt.upper() == 'JPG':
        save_kwargs['quality'] = quality
    pil.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()

# ----------------------- Streamlit UI -----------------------

st.title("🖼️ Image Processing Toolkit — Streamlit + OpenCV")

# Menu bar (Top) simulated using columns
c1, c2, c3 = st.columns([1, 6, 1])
with c1:
    if st.button('Open'):
        st.session_state['open_click'] = True
with c2:
    st.write('')
with c3:
    if st.button('Save'):
        st.session_state['save_click'] = True

# Sidebar: Upload & operations
st.sidebar.header('File')
uploaded = st.sidebar.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg', 'bmp'])
use_camera = st.sidebar.checkbox('Use Camera Input (Streamlit camera_input)', value=False)
if use_camera:
    camera_file = st.camera_input('Take a picture')
    if camera_file is not None and uploaded is None:
        uploaded = camera_file

# operation categories
st.sidebar.header('Operations')
show_info = st.sidebar.checkbox('Show Image Info', value=True)
color_ops = st.sidebar.expander('Color Conversions', expanded=False)
with color_ops:
    color_mode = st.selectbox('Convert to', ['None', 'RGB', 'HSV', 'YCbCr', 'Grayscale'])

trans_ops = st.sidebar.expander('Transformations', expanded=False)
with trans_ops:
    rot_angle = st.slider('Rotation angle (deg)', -180, 180, 0)
    scale_factor = st.slider('Scale factor', 0.1, 3.0, 1.0)
    tx = st.slider('Translate X (pixels)', -200, 200, 0)
    ty = st.slider('Translate Y (pixels)', -200, 200, 0)
    do_affine = st.checkbox('Apply sample Affine', value=False)
    do_perspective = st.checkbox('Apply sample Perspective', value=False)

filter_ops = st.sidebar.expander('Filtering & Morphology', expanded=False)
with filter_ops:
    filter_type = st.selectbox('Filter', ['None', 'Gaussian', 'Median', 'Mean', 'Sobel', 'Laplacian'])
    ksize = st.slider('Kernel size (odd)', 1, 31, 3, step=2)
    morph = st.selectbox('Morphology', ['None', 'Erode', 'Dilate', 'Open', 'Close'])

edge_ops = st.sidebar.expander('Enhancement & Edge Detection', expanded=False)
with edge_ops:
    do_hist_eq = st.checkbox('Histogram Equalization', value=False)
    do_contrast = st.checkbox('Contrast Stretching', value=False)
    do_sharpen = st.checkbox('Sharpen', value=False)
    edge_method = st.selectbox('Edge Detector', ['None', 'Canny', 'Sobel', 'Laplacian'])
    if edge_method == 'Canny':
        low_thr = st.slider('Canny low threshold', 0, 500, 50)
        high_thr = st.slider('Canny high threshold', 0, 500, 150)

compress_ops = st.sidebar.expander('Compression & Save', expanded=False)
with compress_ops:
    save_format = st.selectbox('Save format', ['PNG', 'JPG', 'BMP'])
    jpg_quality = st.slider('JPG Quality', 10, 100, 90)

# Side controls
apply_button = st.sidebar.button('Apply Operations')
reset_button = st.sidebar.button('Reset')

# Session state for storing images
if 'orig' not in st.session_state:
    st.session_state['orig'] = None
if 'proc' not in st.session_state:
    st.session_state['proc'] = None
if 'meta' not in st.session_state:
    st.session_state['meta'] = {}

# Load image
if uploaded is not None:
    img_bgr, meta = read_image(uploaded)
    st.session_state['orig'] = img_bgr
    st.session_state['proc'] = img_bgr.copy()
    st.session_state['meta'] = meta

if reset_button:
    st.session_state['proc'] = st.session_state['orig']

orig = st.session_state.get('orig', None)
proc = st.session_state.get('proc', None)
meta = st.session_state.get('meta', {})

# Apply operations when user clicks
if apply_button and orig is not None:
    proc = orig.copy()

    # Color conversions
    if color_mode != 'None':
        conv = convert_color(proc, color_mode)
        if color_mode == 'Grayscale':
            proc = conv
        else:
            # convert returned array back to BGR for consistent downstream ops
            if color_mode == 'RGB':
                proc = cv2.cvtColor(conv, cv2.COLOR_RGB2BGR)
            elif color_mode == 'HSV':
                proc = cv2.cvtColor(conv, cv2.COLOR_HSV2BGR)
            elif color_mode == 'YCbCr':
                proc = cv2.cvtColor(conv, cv2.COLOR_YCrCb2BGR)

    # Transformations
    if rot_angle != 0:
        proc = rotate_image(proc, rot_angle, scale=1.0)
    if scale_factor != 1.0:
        proc = scale_image(proc, scale_factor, scale_factor)
    if tx != 0 or ty != 0:
        proc = translate_image(proc, tx, ty)
    if do_affine:
        h, w = proc.shape[:2]
        src = [(0,0),(w-1,0),(0,h-1)]
        dst = [(0, int(0.33*h)), (int(0.85*w), int(0.25*h)), (int(0.15*w), int(0.7*h))]
        proc = affine_transform(proc, src, dst)
    if do_perspective:
        h, w = proc.shape[:2]
        src = [(0,0),(w,0),(w,h),(0,h)]
        dst = [(int(0.0*w), int(0.0*h)), (int(0.9*w), int(0.05*h)), (int(0.85*w), int(0.95*h)), (int(0.05*w), int(0.9*h))]
        proc = perspective_transform(proc, src, dst)

    # Filtering
    if filter_type != 'None':
        proc = apply_filter(proc, filter_type, ksize=ksize)

    # Morphology
    if morph != 'None':
        proc = morphology(proc, morph, kernel_size=ksize)

    # Enhancement
    if do_hist_eq:
        proc = histogram_equalization(proc)
    if do_contrast:
        if len(proc.shape) == 2:
            proc = cv2.equalizeHist(proc)
        else:
            proc = contrast_stretching(proc)
    if do_sharpen:
        proc = sharpen_image(proc)

    # Edge detection
    if edge_method != 'None':
        if edge_method == 'Canny':
            edge_img = canny_edge(proc, low_thr, high_thr)
            proc = edge_img
        elif edge_method == 'Sobel':
            proc = apply_filter(proc, 'Sobel', ksize=ksize)
        elif edge_method == 'Laplacian':
            proc = apply_filter(proc, 'Laplacian', ksize=ksize)

    st.session_state['proc'] = proc

# Display Area: two columns
col1, col2 = st.columns(2)
with col1:
    st.subheader('Original Image')
    if orig is None:
        st.info('No image loaded — upload or use camera input from sidebar')
    else:
        st.image(to_pil(orig), use_column_width=True)
with col2:
    st.subheader('Processed Image')
    if proc is None:
        st.info('No processed image yet')
    else:
        # If processed is single channel, ensure PIL handles it
        if len(proc.shape) == 2:
            st.image(Image.fromarray(proc), use_column_width=True)
        else:
            st.image(to_pil(proc), use_column_width=True)

# Status bar / Info
st.markdown('---')
cols = st.columns([1,1,1,1])
info = image_info(orig, meta)
cols[0].metric('Width', info.get('Width', '-'))
cols[1].metric('Height', info.get('Height', '-'))
cols[2].metric('Channels', info.get('Channels', '-'))
# File size calculation
if orig is not None:
    raw_png = save_bytes(orig, fmt='PNG')
    raw_jpg = save_bytes(orig, fmt='JPEG', quality=jpg_quality)
    size_png = len(raw_png)
    size_jpg = len(raw_jpg)
    cols[3].metric('Size PNG (bytes)', f"{size_png}")
    st.write(f"Size JPG (quality={jpg_quality}) = {size_jpg} bytes")

st.markdown('---')

# Save processed image
if proc is not None:
    buf = save_bytes(proc, fmt=save_format, quality=jpg_quality)
    st.download_button('Download Processed Image', data=buf, file_name=f'processed.{save_format.lower()}')

# Bonus: comparison mode (split view)
st.sidebar.header('Bonus')
if st.sidebar.checkbox('Split Comparison Mode (half original/half processed)') and orig is not None and proc is not None:
    # create split image
    ow, oh = meta.get('size', (orig.shape[1], orig.shape[0]))
    # resize processed to original size (if needed)
    if proc.shape[:2] != orig.shape[:2]:
        proc_r = cv2.resize(proc, (orig.shape[1], orig.shape[0]))
    else:
        proc_r = proc
    # ensure both are 3-channel
    if len(proc_r.shape) == 2:
        proc_r = cv2.cvtColor(proc_r, cv2.COLOR_GRAY2BGR)
    left = orig.copy()
    right = proc_r.copy()
    split = left.copy()
    split[:, :split.shape[1]//2] = left[:, :split.shape[1]//2]
    split[:, split.shape[1]//2:] = right[:, split.shape[1]//2:]
    st.subheader('Split Comparison (left=original, right=processed)')
    st.image(to_pil(split), use_column_width=True)

# Simple real-time webcam mode: apply a selected filter to the camera frame
if st.sidebar.checkbox('Webcam Live Mode'):
    st.info('Webcam Live Mode (experimental). Press Stop to end.')
    run = st.checkbox('Run Webcam')
    FRAME_PLACEHOLDER = st.empty()
    cap = None
    try:
        if run:
            cap = cv2.VideoCapture(0)
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.warning('No camera frame available')
                    break
                # apply a simple selected op
                display = frame
                if filter_type != 'None':
                    display = apply_filter(display, filter_type, ksize=ksize)
                    if len(display.shape) == 2:
                        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
                FRAME_PLACEHOLDER.image(to_pil(display), use_column_width=True)
                run = st.checkbox('Run Webcam', value=True)
        else:
            FRAME_PLACEHOLDER.empty()
    finally:
        if cap is not None:
            cap.release()

st.caption('Built with Streamlit + OpenCV — includes color conversions, transforms, filters, morphology, enhancement, edge detection, and compression examples.')


