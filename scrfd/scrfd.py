import numpy as np
import tritonclient.http as httpclient
import cv2

def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)



def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def scrfd_preproc(image: np.ndarray, output_size: tuple):
    '''Функция предобработки изображения перед подачей в нейросеть SCRFD. Исходное изображение будет
    растянуто или сжато, чтобы полностью поместиться в окно (прямоугольник) размером output_size,
    свободные пиксели будут заполнены нулями.
    Входные параметры:
    image: np.ndarray - исходное изображение в формате (H, W, C) 0..255
    output_size: tuple - размерность, к которой будет приведено исходное изображение (например (320, 320)),
    значение должно соответствовать формату, который принимает нейросеть
    Возвращаемые значения:
    prepared_image: np.ndarray - изображение, подготовленное к подаче в нейросеть в формате (B, C, H, W)
    в данной версии размер батча всегда равен 1
    det_scale: float - коэффициент шкалирования нового изображения (насколько мы его в данной функции
    растянули или сжали, чтобы он мог быть подан в нейронку)'''

    im_ratio = float(image.shape[0]) / image.shape[1]
    model_ratio = float(output_size[1]) / output_size[0]

    if im_ratio > model_ratio:
        new_height = output_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = output_size[0]
        new_height = int(new_width * im_ratio)

    det_scale = float(new_height) / image.shape[0]
    resized_img = cv2.resize(image, (new_width, new_height))
    det_img = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = resized_img
    output_size = tuple(det_img.shape[0:2][::-1])
    prepared_image = cv2.dnn.blobFromImage(det_img, 1.0 / 128, output_size, (127.5, 127.5, 127.5), swapRB=True)

    return prepared_image, det_scale



def scrfd_postproc(inputs: list, feat_stride_fpn: list, fmc: int, num_anchors: int, thresh: float, 
    nms_thresh: float, use_kps: bool, det_scale: float, preproc_image_heigh: int, preproc_image_width: int, max_num: int = 0):
    '''Функция постобработки выхода нейросети SCRFD. Реализует в себе логику FPN и NMS
    Входные параметры:
    inputs: list - список np.ndarray массивов, которые получены в результате работы детектора SCRFD
    feat_stride_fpn: list[int] - список параметров FPN, которые жестко заданы для конкретной tensorrt модели
    fmc: int
    num_anchors: int
    thresh: float - порог для фильтрации bbox'ов с низким скором
    nms_thresh: float - порог для работы NON MAXIMUM SUPRESSION
    use_kps: bool - возвращать ли keypoints
    det_scale: float - коэффициент шкалирования исходного изображения
    preproc_image_heigh: int - высота изображения, которое подавалось в нейросеть
    preproc_image_width: int - ширина изображения, которое подавалось в нейросеть
    max_num: int
    Возвращаемые значения:
    det: np.ndarray - координаты bbox'ов в формате (число детекций, 5), 5: (left, top, right, bottom, score)
    kpss: np.ndarray - ключевые точки в формате (число детекций, 5, 2), 5 - количество ключевых точек на детекцию, 
    2 - количество координат на ключевую точку'''
    
    scores_list, bboxes_list, kpss_list = [], [], []

    for idx, stride in enumerate(feat_stride_fpn):
        scores = inputs[idx * fmc][0]
        bbox_preds = inputs[idx * fmc + 1][0] * stride
        kps_preds = inputs[idx * fmc + 2][0] * stride
        height = preproc_image_heigh // stride
        width = preproc_image_width // stride
        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(
                np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if num_anchors > 1:
            anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
        pos_inds = np.where(scores >= thresh)[0]
        bboxes = distance2bbox(anchor_centers, bbox_preds)
        pos_scores = scores[pos_inds]
        pos_bboxes = bboxes[pos_inds]
        scores_list.append(pos_scores)
        bboxes_list.append(pos_bboxes)

        kpss = distance2kps(anchor_centers, kps_preds)
        kpss = kpss.reshape((kpss.shape[0], -1, 2))
        pos_kpss = kpss[pos_inds]
        kpss_list.append(pos_kpss)


    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]
    bboxes = np.vstack(bboxes_list) / det_scale
    if use_kps:
        kpss = np.vstack(kpss_list) / det_scale
    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
    pre_det = pre_det[order, :]
    keep = nms(pre_det, nms_thresh)
    det = pre_det[keep, :]
    if use_kps:
        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]
    else:
        kpss = None
    if max_num > 0 and det.shape[0] > max_num:
        area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
        img_center = img.shape[0] // 2, img.shape[1] // 2
        offsets = np.vstack([
            (det[:, 0] + det[:, 2]) / 2 - img_center[1],
            (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
        if metric == 'max':
            values = area
        else:
            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
        
        bindex = np.argsort(values)[::-1]  # some extra weight on the centering
        bindex = bindex[0:max_num]
        det = det[bindex, :]
        if kpss is not None:
            kpss = kpss[bindex, :]

    return det, kpss