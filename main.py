import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# =========================
# Model Architecture
# =========================

class CNNModel(nn.Module):

    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))


# =========================
# Class Names
# =========================

class_names = [
    'F_Banana',
    'F_Lemon',
    'F_Lulo',
    'F_Mango',
    'F_Orange',
    'F_Strawberry',
    'F_Tamarillo',
    'F_Tomato',
    'S_Banana',
    'S_Lemon',
    'S_Lulo',
    'S_Mango',
    'S_Orange',
    'S_Strawberry',
    'S_Tamarillo',
    'S_Tomato'
]

# =========================
# Load Model Once
# =========================

model = CNNModel(num_classes=16)

model.load_state_dict(
    torch.load(r"C:\code\fruit_freshness_indicator_DLproject\final_tuned_model.pth", map_location="cpu")
)

model.eval()

# =========================
# Same Inference Transform
# =========================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# Prediction Function
# =========================

def predict_image(image):

    image = image.convert("RGB")

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.softmax(output, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]

    return predicted_class, confidence.item() * 100
