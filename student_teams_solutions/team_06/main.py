import os

import numpy as np
import pandas as pd
import cv2 as cv
from tqdm import tqdm
from tensorflow import keras

import src.bounfing_box_utils as bounding_box

DIR_NAME_DECTECTION = "your_name(same used than in yolo prediction)"
IMAGES_PATH = "your_image_to_test_path"
DETECT_DIR = "./models/yolov5/runs/detect/"+DIR_NAME_DECTECTION+"/labels/"



results_df = bounding_box.predict_pipeline(IMAGE_PATH, DETECT_DIR)


car_models_footprint_file_dir = "./datasets/car_models_footprint.csv"
test_path                     = "./datasets/datasets_test/test/"

car_models_footrpint = pd.read_csv(car_models_footprint_file_dir, sep=";")

im_size = (240, 240)

model = keras.models.load_model('./models/modelMN2/')

def resize_image(img, im_size=im_size):
    return cv.resize(img, im_size,interpolation =cv.INTER_LINEAR)
    
encoding = {0: 'AM General Hummer SUV 2000',
 1: 'Acura TL Type-S 2008',
 2: 'Acura ZDX Hatchback 2012',
 3: 'Aston Martin V8 Vantage Convertible 2012',
 4: 'Aston Martin V8 Vantage Coupe 2012',
 5: 'Aston Martin Virage Convertible 2012',
 6: 'Aston Martin Virage Coupe 2012',
 7: 'Audi 100 Sedan 1994',
 8: 'Audi 100 Wagon 1994',
 9: 'Audi A5 Coupe 2012',
 10: 'Audi S4 Sedan 2007',
 11: 'Audi S5 Convertible 2012',
 12: 'Audi S5 Coupe 2012',
 13: 'Audi TT RS Coupe 2012',
 14: 'Audi TTS Coupe 2012',
 15: 'Audi V8 Sedan 1994',
 16: 'BMW 1 Series Convertible 2012',
 17: 'BMW 1 Series Coupe 2012',
 18: 'BMW 3 Series Wagon 2012',
 19: 'BMW ActiveHybrid 5 Sedan 2012',
 20: 'BMW M5 Sedan 2010',
 21: 'BMW X6 SUV 2012',
 22: 'Bentley Continental GT Coupe 2007',
 23: 'Buick Rainier SUV 2007',
 24: 'Buick Regal GS 2012',
 25: 'Cadillac SRX SUV 2012',
 26: 'Chevrolet Camaro Convertible 2012',
 27: 'Chevrolet Cobalt SS 2010',
 28: 'Chevrolet Corvette ZR1 2012',
 29: 'Chevrolet Express Van 2007',
 30: 'Chevrolet HHR SS 2010',
 31: 'Chevrolet Malibu Sedan 2007',
 32: 'Chevrolet Silverado 1500 Classic Extended Cab 2007',
 33: 'Chevrolet Silverado 1500 Extended Cab 2012',
 34: 'Chevrolet TrailBlazer SS 2009',
 35: 'Chrysler 300 SRT-8 2010',
 36: 'Chrysler Crossfire Convertible 2008',
 37: 'Chrysler PT Cruiser Convertible 2008',
 38: 'Dodge Caliber Wagon 2007',
 39: 'Dodge Caliber Wagon 2012',
 40: 'Dodge Charger Sedan 2012',
 41: 'Dodge Durango SUV 2007',
 42: 'Dodge Magnum Wagon 2008',
 43: 'Eagle Talon Hatchback 1998',
 44: 'FIAT 500 Convertible 2012',
 45: 'Ferrari 458 Italia Convertible 2012',
 46: 'Ferrari 458 Italia Coupe 2012',
 47: 'Fisker Karma Sedan 2012',
 48: 'Ford E-Series Wagon Van 2012',
 49: 'Ford Expedition EL SUV 2009',
 50: 'Ford F-150 Regular Cab 2007',
 51: 'Ford F-450 Super Duty Crew Cab 2012',
 52: 'Ford Fiesta Sedan 2012',
 53: 'Ford Focus Sedan 2007',
 54: 'Ford Freestar Minivan 2007',
 55: 'Ford GT Coupe 2006',
 56: 'Ford Mustang Convertible 2007',
 57: 'Ford Ranger SuperCab 2011',
 58: 'GMC Yukon Hybrid SUV 2012',
 59: 'HUMMER H3T Crew Cab 2010',
 60: 'Honda Accord Coupe 2012',
 61: 'Honda Odyssey Minivan 2012',
 62: 'Hyundai Azera Sedan 2012',
 63: 'Hyundai Genesis Sedan 2012',
 64: 'Hyundai Sonata Hybrid Sedan 2012',
 65: 'Hyundai Sonata Sedan 2012',
 66: 'Hyundai Veracruz SUV 2012',
 67: 'Infiniti G Coupe IPL 2012',
 68: 'Infiniti QX56 SUV 2011',
 69: 'Jaguar XK XKR 2012',
 70: 'Jeep Grand Cherokee SUV 2012',
 71: 'Jeep Liberty SUV 2012',
 72: 'Jeep Wrangler SUV 2012',
 73: 'Lamborghini Aventador Coupe 2012',
 74: 'Lamborghini Gallardo LP 570-4 Superleggera 2012',
 75: 'Lamborghini Reventon Coupe 2008',
 76: 'Land Rover LR2 SUV 2012',
 77: 'Maybach Landaulet Convertible 2012',
 78: 'Mercedes-Benz 300-Class Convertible 1993',
 79: 'Mercedes-Benz SL-Class Coupe 2009',
 80: 'Mercedes-Benz Sprinter Van 2012',
 81: 'Mitsubishi Lancer Sedan 2012',
 82: 'Nissan Leaf Hatchback 2012',
 83: 'Plymouth Neon Coupe 1999',
 84: 'Porsche Panamera Sedan 2012',
 85: 'Rolls-Royce Ghost Sedan 2012',
 86: 'Rolls-Royce Phantom Drophead Coupe Convertible 2012',
 87: 'Rolls-Royce Phantom Sedan 2012',
 88: 'Scion xD Hatchback 2012',
 89: 'Spyker C8 Coupe 2009',
 90: 'Suzuki Aerio Sedan 2007',
 91: 'Suzuki Kizashi Sedan 2012',
 92: 'Suzuki SX4 Sedan 2012',
 93: 'Toyota Camry Sedan 2012',
 94: 'Toyota Corolla Sedan 2012',
 95: 'Toyota Sequoia SUV 2012',
 96: 'Volkswagen Beetle Hatchback 2012',
 97: 'Volvo 240 Sedan 1993',
 98: 'Volvo C30 Hatchback 2012',
 99: 'Volvo XC90 SUV 2007'}

y_pred = []
img_names_test = []

for img_name in os.listdir(test_path) :
    if img_name[-3:] == 'jpg':
        try: 
            car_model_name = str(img_name).split("_")[0]

            img = cv.imread(test_path + img_name)
            img = resize_image(img, im_size=(240, 240))
            img = img / 255

            pred = model.predict(np.array([img]))
            car_model = encoding[np.argmax(pred)]
            pred_co2 = car_models_footrpint[car_models_footrpint['models'] == car_model]['Average of CO2 (g per km)'].values[0]

            y_pred.append(pred_co2)
            img_names_test.append(img_name)
            
        except:
            img_names_test.append(img_name)
            y_pred.append(0)

data = {
    'im_name' : img_names_test,
    'e'       : y_pred
}


pred_test = pd.DataFrame(data)
# This may not work (not tested)
result_test = results_df.join(pred_test, on="im_name" )
result_test.to_csv('submit_emission.csv')
