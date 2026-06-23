from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import time
import torch
import torchvision.models as models
from torchvision import transforms

app = FastAPI(title="Leaf Disease API")

# Load model once at startup

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 6)

checkpoint = torch.load(
"models/ResNet18_checkpoint_epoch3.pth",
map_location="cpu"
)

model.load_state_dict(checkpoint["model_state"])
model.eval()

class_names = {
0: "Pepper,_bell___Bacterial_spot",
1: "Pepper,_bell___healthy",
2: "Tomato___Bacterial_spot",
3: "Tomato___Early_blight",
4: "Tomato___healthy",
5: "Tomato___Late_blight"
}

preprocess = transforms.Compose([
transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485,0.456,0.406],
std=[0.229,0.224,0.225]
)
])

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.perf_counter()

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    x = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)

    confidence, pred = probs.max(dim=1)

    inference_ms = (
        time.perf_counter() - start_time
    ) * 1000

    return {
        "class": class_names[pred.item()],
        "confidence": round(
            confidence.item() * 100,
            2
        ),
        "inference_ms": round(
            inference_ms,
            2
        )
    }

