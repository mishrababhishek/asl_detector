from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch import device, no_grad
from torch.cuda import is_available


class SignLanguageDetector:
    def __init__(self):
        self._device = device("cuda" if is_available() else "cpu")
        self._processor = AutoImageProcessor.from_pretrained("RavenOnur/Sign-Language")
        self._model = AutoModelForImageClassification.from_pretrained(
            "RavenOnur/Sign-Language"
        )
        self._model.to(self._device)
        self._model.eval()

    def classify(self, image):
        with no_grad():
            inputs = self._processor(image, return_tensors="pt").to(self._device)
            logits = self._model(**inputs).logits
            predicted_label = logits.argmax(-1).item()
            return self._model.config.id2label[predicted_label]
