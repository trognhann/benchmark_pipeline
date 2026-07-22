import os, time
from PIL import Image, ImageDraw
import numpy as np
try:
    from retinaface_ import cfg_mnet
    from retinaface_.prior_box import PriorBox
    from retinaface_.py_cpu_nms import py_cpu_nms
    from retinaface_.box_utils import decode, decode_landm
except:
    from .retinaface_ import cfg_mnet
    from .retinaface_.prior_box import PriorBox
    from .retinaface_.py_cpu_nms import py_cpu_nms
    from .retinaface_.box_utils import decode, decode_landm

import assets_bin
from config import AES_de

import onnxruntime

ort_sess_options = onnxruntime.SessionOptions()
ort_sess_options.intra_op_num_threads = int(os.environ.get('ort_intra_op_num_threads', 0))

# Cache the decrypted model to disk to skip AES decryption on subsequent runs
_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_model", "face_det_retina.onnx")

if os.path.exists(_CACHE_PATH):
    torch_mtcnn = onnxruntime.InferenceSession(_CACHE_PATH, sess_options=ort_sess_options)
else:
    _model_bytes = AES_de(assets_bin.faceDet, key=AES_de(assets_bin.Author).decode("utf-8"))
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as _f:
        _f.write(_model_bytes)
    torch_mtcnn = onnxruntime.InferenceSession(_CACHE_PATH, sess_options=ort_sess_options)
    del _model_bytes

cfg = cfg_mnet

# Expand the area of the detected face frame by margin pixels in proportion to the face frame;
# expand the avatar area frame according to a fixed aspect ratio
def margin_face(box, img_HW, margin=0.5):
    x1, y1, x2, y2 = [c for c in box]
    w, h = x2 - x1, y2 - y1
    new_x1 = max(0, x1 - margin*w)
    new_x2 = min(img_HW[1], x2 + margin * w)
    x_d = min(x1-new_x1, new_x2-x2)
    new_w = x2 -x1 + 2 * x_d  # Make sure that the left and right sides of the face are expanded by the same x_d pixels
    new_x1 = x1-x_d
    new_x2 = x2+x_d
    new_h = 1. * new_w   # Image (112*112) aspect ratio is 1.0
    if new_h>=h:
        y_d = new_h-h  # # Make sure that both sides of the face are extended by the same half of y_d pixels
        new_y1 = max(0, y1 - y_d//2)
        new_y2 = min(img_HW[0], y2 + y_d//2)
    else:
        y_d = abs(new_h - h)  #  Make sure that both sides of the face are reduced by half the pixels of the same y_d
        new_y1 = max(0, y1 + y_d // 2)
        new_y2 = min(img_HW[0], y2 - y_d // 2)
    # Since the image portrait may be close to the edge of the photo,
    # it is very likely that it will not be able to expand if it extends to the edge.
    # Therefore, the width always expands the same on the left and right,
    # but the height may not necessarily expand in a ratio of 1.0 to the relative width.
    return list(map(int, [new_x1, new_y1, new_x2, new_y2]))


def detect_face(img, resize=1, confidence_threshold=0.8, top_k=10, nms_threshold=0.3, keep_top_k=5):
    img = np.float32(img)
    im_height, im_width, _ = img.shape
    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    # Whether to scale the entire image input proportionally
    img -= (123, 117, 104)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    ort_inputs = {torch_mtcnn.get_inputs()[0].name: img}
    ort_outs = torch_mtcnn.run(None, ort_inputs)
    loc, conf, landms = ort_outs
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    prior_data = priorbox.forward()

    boxes = decode(loc.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale * resize
    scores = conf.squeeze(0)[:, 1]
    landms = decode_landm(landms.squeeze(0), prior_data, cfg['variance'])
    scale1 = np.array([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    landms = landms * scale1 * resize
    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]
    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]
    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]
    # keep top-K faster NMS
    dets = dets[:keep_top_k, :4]
    landms = landms[:keep_top_k, :]
    # dets = np.concatenate((dets, landms), axis=1)
    box_order = np.argsort((dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1]))[::-1]
    dets = dets[box_order, :]
    landms = landms[box_order, :]
    landms = np.reshape(landms, (landms.shape[0], 5, 2))
    if 0 in dets.shape:
        return None, None
    return dets, landms

if __name__ == "__main__":
    
    img_path = r"C:\Users\ASUS ZENBOOK\Downloads\ảnh 1.png"
    img = Image.open(img_path)
    dets, landms = detect_face(img)
    print("Boxes:\n", dets)
    print("Landmarks:\n", landms)
    
    if dets is not None:
        draw = ImageDraw.Draw(img)
        for i, box in enumerate(dets):
            # Lấy tọa độ bounding box
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            
            # Vẽ các điểm landmarks (5 điểm)
            if landms is not None:
                for pt in landms[i]:
                    x, y = pt
                    d = 3  # Kích thước của dấu X
                    draw.line((x-d, y-d, x+d, y+d), fill="yellow", width=2)
                    draw.line((x-d, y+d, x+d, y-d), fill="yellow", width=2)
        
        img.show()
