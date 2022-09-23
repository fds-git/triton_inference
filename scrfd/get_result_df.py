import pickle
import os
import pandas as pd

images_path = './destination/FaceMaskDetectionDataset/images_clean/'

with open('./destination/FaceMaskDetectionDataset/landmarks.pkl', 'rb') as f:
    landmarks_arr = pickle.load(f)

with open('./destination/FaceMaskDetectionDataset/image_paths.pkl', 'rb') as f:
    image_names = pickle.load(f)

# Получаем названия файлов, прошедших ручной отбор по качеству ландмарков
filtered_dataset_path = './destination/FaceMaskDetectionDataset/images_landmark_filtered/'
filtered_image_names = os.listdir(filtered_dataset_path)

# Загружаем исходный датафрейм с информацией о всех изображениях
dataframe = pd.DataFrame({'image_names': image_names, 'landmarks': landmarks_arr})

# Получаем итоговый датафрейм с информацией только об отобранных изображениях
result_dataframe = dataframe[dataframe['image_names'].isin(filtered_image_names)]
result_dataframe = result_dataframe.reset_index(drop=True)

result_dataframe.to_pickle('./destination/FaceMaskDetectionDataset/mask_dataframe.pkl')