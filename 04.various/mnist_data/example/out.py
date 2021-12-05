#import argparse
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

model_dir = os.path.join(os.getcwd(), "..")

with open(os.path.join(model_dir, "signature.json"), "r",encoding="utf-8_sig") as f:
    signature = json.load(f)

inputs = signature.get("inputs")
outputs = signature.get("outputs")

model = tf.saved_model.load(tags=signature.get("tags"), export_dir=model_dir)
predict_fn = model.signatures["serving_default"]    

image = Image.open('img.png')

if image.mode != "RGB":
    image = image.convert("RGB")

input_width, input_height = inputs["Image"]["shape"][1:3]
image = image.resize((input_width, input_height))
image = np.asarray(image) / 255.0
image= np.expand_dims(image, axis=0).astype(np.float32)

feed_dict = {}
feed_dict[list(inputs.keys())[0]] = tf.convert_to_tensor(image)
outputs = predict_fn(**feed_dict)

out_keys = ["label", "confidence"]
results = {}
for key, tf_val in outputs.items():
    val = tf_val.numpy().tolist()[0]
    if isinstance(val, bytes):
        val = val.decode()
    results[key] = val
confs = results["Confidences"]
labels = signature.get("classes").get("Label")
output = [dict(zip(out_keys, group)) for group in zip(labels, confs)]
sorted_output = max(output, key=lambda k: k["confidence"])

out=[]
out.extend(sorted_output.values())

print(out[0])
