import base64
import requests
import cv2
import numpy as np
import time

try:
    # Create a dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite('dummy.jpg', img)

    data = {
        'distance': 100,
        'ss': 1,
        'at': 1,
        'mobility': 2,
        'layer': 1
    }

    files = {
        'image': ('dummy.jpg', open('dummy.jpg', 'rb'), 'image/jpeg')
    }

    response = requests.post('http://127.0.0.1:5000/api/transmit', data=data, files=files)
    print("Status:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("Error:", e)
