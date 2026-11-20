import cv2
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_NAME = "raxtemur/trocr-base-ru"

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется: {device}")
model.to(device)
model.eval()

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Файл '{image_path}' не найден!")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb_img)

def recognize_handwriting(image_path):
    pil_image = preprocess_image(image_path)
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=64)
        
    recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return recognized_text

if __name__ == "__main__":
    test_image = "word4.png"
    result = recognize_handwriting(test_image)
    
    print("\n==============================")
    print(f"Распознанный текст: {result}")
    print("==============================")

