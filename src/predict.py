import torch
from PIL import Image
from torchvision import transforms
from models import ConvNN

names = [str(i) for i in range(10)]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

modell = ConvNN()
modell.load_state_dict(torch.load('../models/cnn_model.pt', weights_only=True))
modell.eval()

def predict(inp):
    # Get NumPy array from dict
    image_np = inp['composite']  # ← THIS is the correct key!
    transform = transforms.Compose([
        transforms.ToPILImage(mode='RGBA'),  # ← Explicit RGBA
        transforms.Grayscale(num_output_channels=1),  # ← RGBA → Grayscale (1 channel)
        transforms.Resize((28, 28)),  # ← Already 28x28, but safe
        transforms.ToTensor(),  # ← PIL → Tensor (1, 28, 28)
        transforms.Lambda(lambda x: 1.0 - x),  # ← Invert (black bg, white digits)
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Optional: Invert colors (MNIST expects white digits on black bg; sketchpad is often black on white)
    # Add this to transform if your predictions are wrong: transforms.Lambda(lambda x: 1 - x),

    test_imgT = transform(image_np).unsqueeze(0).to(device)

    with torch.no_grad():
        modell.eval()
        outputs = modell(test_imgT)
        probs = torch.softmax(outputs, dim=1)
        _, prediction = torch.max(probs, dim=1)
        return f"Predicted digit: {names[int(prediction.item())]}"