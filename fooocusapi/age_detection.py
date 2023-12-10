# coding=utf-8

from io import BytesIO

from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification


def predict_age(bytes_data) -> str:
    try:
        im = Image.open(BytesIO(bytes_data))

        # Init model, transforms
        model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
        transforms = ViTImageProcessor.from_pretrained('nateraw/vit-age-classifier')

        # Transform our image and pass it through the model
        inputs = transforms(im, return_tensors='pt')
        output = model(**inputs)

        # Predicted Class probabilities
        proba = output.logits.softmax(1)

        # Predicted Classes
        preds = proba.argmax(1)
        return str(model.config.id2label[preds.item()]).replace('10', '17')
    except Exception as e:
        print(f"predict_age error: {e}")
        return "20-26"
