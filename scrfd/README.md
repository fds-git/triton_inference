## Модель детекции лиц и ключевых точек, используемая для разметки изображений (получения датасета ключевых точек)
- source - папка с исходным датасетом https://www.kaggle.com/datasets/wobotintelligence/face-mask-detection-dataset
- destination - папка с размеченным датасетом
- models - папка с моделью scrfd в формате .txt
- pred_landmarks.py - скрипт для авторазметки
- source_config.yaml - конфиги

### Порядок работы:
#### Запустить сервер с моделью:
	docker run --rm --gpus=1 -p8000:8000 -p8001:8001 -p8002:8002 -v /home/dima/Work/triton_inference/scrfd/models:/models nvcr.io/nvidia/tritonserver:21.07-py3 tritonserver --model-repository=/models

#### Запустить клиент с пробросом директории с изображениями и скриптами
	docker run -it --rm --net=host -v /home/dima/Work/triton_inference/scrfd/:/home/dima/Work/  nvcr.io/nvidia/tritonserver:21.07-py3-sdk

#### Установить необходимые библиотеки в клиент
	pip install opencv-python
	pip install albumentations

#### Перейти в рабочий каталог
	cd /home/dima/Work/

#### Запустить скрипт, производящий предсказания
	python pred_landmarks.py

После вызова pred_landmarks.py в destination/имя_датасета/ будут созданы 2 папки: images_clean, images_landmark. В 1-й - задетектированные и обрезанные изображения лиц людей, во 2-й - те же изображения с нанесенными на них ключевыми точками. Выходим из контейнеров. 

#### Изменить права на доступ к папкам
	sudo chmod 777 destination/имя_датасета/
	sudo chmod 777 destination/имя_датасета/images_clean/

Далее копируем images_landmark и переименовываем ее в images_landmark_filtered. В этой папке руками удаляем изображения с невалидными ключевыми точками.

#### Запускаем скрипт получения итогового датафрейма
	python get_result_df.py

Итог: в destination/имя_датасета/ папка с изображениями images_clean и датафрейм с именами изображений и координатами меток 