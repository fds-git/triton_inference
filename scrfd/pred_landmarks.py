import time
import numpy as np
import tritonclient.http as httpclient
import cv2
import os
import albumentations as A
import shutil
import pickle
import yaml
from scrfd import scrfd_preproc, scrfd_postproc
from tritonclient.http import InferenceServerClient


def test_infer(triton_client: InferenceServerClient,
                model_name: str,
                input_data: np.ndarray,
                headers=None,
                request_compression_algorithm=None,
                response_compression_algorithm=None):
    '''Функция получения результата работы нейронной сети, развернутой на сервере triton
    Входные параметры:
    triton_client: объект клиента triton server
    model_name: str - название модели, которая хранится на сервере и будет вызываться
    input_data: np.ndarray - входной тензор необходимой для модели размерности (1, 3, 320, 320)
    headers - какие-то заголовки
    request_compression_algorithm: object - алгоритм сжатия (польза только при больших данных)
    response_compression_algorithm: object - алгоритм сжатия (польза только при больших данных)
    Возвращаемые значения:
    results: object - ответ от сервера'''

    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput(name='input.1', shape=input_data.shape, datatype="FP32"))

    # Initialize the data (binary_data обязательно true для быстродействия)
    inputs[0].set_data_from_numpy(input_tensor=input_data, binary_data=True)

    # Если binary_data=True, содержимое тензоров получаем через test_infer("scrfd", input_data).as_numpy('bbox_16')
    # иначе test_infer("res18", input_data).get_response() (то есть в теле json)
    outputs.append(httpclient.InferRequestedOutput('bbox_16', binary_data=True))
    outputs.append(httpclient.InferRequestedOutput('bbox_8', binary_data=True))
    outputs.append(httpclient.InferRequestedOutput('bbox_32', binary_data=True))

    outputs.append(httpclient.InferRequestedOutput('kps_8', binary_data=True))
    outputs.append(httpclient.InferRequestedOutput('kps_16', binary_data=True))
    outputs.append(httpclient.InferRequestedOutput('kps_32', binary_data=True))

    outputs.append(httpclient.InferRequestedOutput('score_16', binary_data=True))
    outputs.append(httpclient.InferRequestedOutput('score_8', binary_data=True))
    outputs.append(httpclient.InferRequestedOutput('score_32', binary_data=True))

    query_params = {'test_1': 1}

    results = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
        query_params=query_params,
        headers=headers,
        request_compression_algorithm=request_compression_algorithm,
        response_compression_algorithm=response_compression_algorithm)

    return results

#--------------------------------------------

# Параметры для предобработки
# Размер изображения, с которым может работать нейронка
height = 320
width = 320
output_size = (height, width)

#--------------------------------------------

# Параметры для нейронки
# Имя нейронки в соответствии с tritonserver
model_name = 'scrfd'

#--------------------------------------------

# Параметры для постобработки
# Шаг fpn - зависит от конкретной нейронки
feat_stride_fpn = [8, 16, 32]
fmc = 3
num_anchors = 2

# Порог для работы NMS
nms_thresh = 0.58
# Нужны ли keypoints
use_kps = True
max_num = 0

#--------------------------------------------

# Параметры отрисовки bbox
LINE_THICKNESS = 2
BLUE_COLOR = (255, 0, 0)

# Загружаем параметры из конфигурационного файла
with open(r'source_config.yaml') as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

# Пути исходного тестового изображения и сохранения изображения,
# с отрисованными bbox и keypoints
source_image_path = configs['source_image_path']

#dest_image_clean_path = configs['dest_image_clean_path']
#dest_image_landmark_path = configs['dest_image_landmark_path']

dest_image_path = configs['dest_image_path']
dest_image_clean_path = dest_image_path + 'images_clean/'
dest_image_landmark_path = dest_image_path + 'images_landmark/'

# Порог для фильтрации bbox по значению score
thresh = configs['thresh']
face_shape = configs['face_shape']

if os.path.isdir(dest_image_clean_path):
    shutil.rmtree(dest_image_clean_path)
os.makedirs(dest_image_clean_path)

if os.path.isdir(dest_image_landmark_path):
    shutil.rmtree(dest_image_landmark_path)
os.makedirs(dest_image_landmark_path)

image_names = os.listdir(source_image_path)

#--------------------------------------------

transform = A.Compose([
    A.Resize(*face_shape),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, angle_in_degrees=True))

triton_client = httpclient.InferenceServerClient(url='localhost:8000', verbose=False, concurrency=20)

pred_timings = []
net_timings = []
post_timings = []

landmarks_arr = []
image_paths = []

face_counter = 0

start_time = time.time()
for image_name in image_names:
    image = cv2.imread(source_image_path + image_name)
    rgb_frame = image.copy()

    #---------------------------Предобработка---------------------------

    start = time.time()
    preproc_image, det_scale = scrfd_preproc(image=rgb_frame, output_size=output_size)
    stop = time.time()
    pred_timings.append(stop - start)

    #--------------------------Вызов нейронки----------------------------

    start = time.time()

    # Если выход бинаризованный
    response = test_infer(triton_client=triton_client, model_name=model_name, input_data=preproc_image)
    # Так как данные передаются в бинарном виде, надо их декодировать
    bbox_16 = response.as_numpy('bbox_16')
    bbox_8 = response.as_numpy('bbox_8')
    bbox_32 = response.as_numpy('bbox_32')

    kps_8 = response.as_numpy('kps_8')
    kps_16 = response.as_numpy('kps_16')
    kps_32 = response.as_numpy('kps_32')

    score_16 = response.as_numpy('score_16')
    score_8 = response.as_numpy('score_8')
    score_32 = response.as_numpy('score_32')

    # Упаковываем именно в таком порядке в соответствии с scrfd.py
    # list np.ndarray (1 3200 1, 1 3200 4, 1 3200 10, 1 800 1, 1 800 4, 1 800 10, 1 200 1, 1 200 4, 1 200 10)
    scrfd_outputs = [score_8, bbox_8, kps_8, score_16, bbox_16, kps_16, score_32, bbox_32, kps_32]

    stop = time.time()
    net_timings.append(stop - start)
     
    #-------------------------Постобработка-------------------

    start = time.time()

    params = {
    'inputs': scrfd_outputs,
    'feat_stride_fpn': feat_stride_fpn,
    'fmc': fmc,
    'num_anchors': num_anchors,
    'thresh': thresh,
    'nms_thresh': nms_thresh,
    'use_kps': use_kps,
    'max_num': max_num,
    'det_scale': det_scale,
    'preproc_image_heigh': preproc_image.shape[2],
    'preproc_image_width': preproc_image.shape[3]
    }

    bboxes, landmarkss = scrfd_postproc(**params)

    stop = time.time()
    post_timings.append(stop - start)

    if bboxes.shape[0] != 0:
        for box, landmarks in zip(bboxes, landmarkss):
            left, top, right, bottom, prob = box[0], box[1], box[2], box[3], box[4]

            # Если координаты bbox выходят за рамки изображения,
            # корректируем bbox (иначе ошибка albumintation)
            if left < 0:
                left = 0
            if right > image.shape[1]:
                right = image.shape[1]
            if top < 0:
                top = 0
            if bottom > image.shape[0]:
                bottom = image.shape[0]

            #if (bottom - top < 50) or (right - left < 50):
            #    continue

            face_image = image[int(top): int(bottom), int(left): int(right)]
            new_landmarks = landmarks - np.array([[left, top]])
            face_counter += 1
            transformed = transform(image=face_image, keypoints=new_landmarks)
            transformed_image = transformed['image']
            transformed_landmarks = np.array(transformed['keypoints'])

            # Сохраняем исходное изображение без keypoints
            cv2.imwrite(dest_image_clean_path + str(face_counter) + '_' + image_name, transformed_image)

            #  Отрисовываем keypoints
            for transformed_landmark in transformed_landmarks:
                cv2.circle(transformed_image, (int(transformed_landmark[0]), int(transformed_landmark[1])), 1, (0, 0, 255), -1)

            # Сохраняем исходное изображение с keypoints
            cv2.imwrite(dest_image_landmark_path + str(face_counter) + '_' + image_name, transformed_image)

            # Сохраняем информацию о названиях изображений и метках
            landmarks_arr.append(transformed_landmarks)
            image_paths.append(str(face_counter) + '_' + image_name)


# Лучше не сохранять через pandas, т.к. из-за различия
# версий могут быть проблемы
with open(dest_image_path + 'landmarks.pkl', 'wb') as f:
    pickle.dump(landmarks_arr, f)

with open(dest_image_path + 'image_paths.pkl', 'wb') as f:
    pickle.dump(image_paths, f)

end_time = time.time()

print(f"Pred time: {np.sum(pred_timings)}")
print(f"Net time: {np.sum(net_timings)}")
print(f"Post time: {np.sum(post_timings)}")
print(f"Общее время: {end_time - start_time}")