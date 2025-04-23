import os
import torch
import torch.nn as nn
from telegram import Update
from telegram.ext import Application, MessageHandler, filters
from PIL import Image
from torchvision import transforms

# Load model
model = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(31752, 3)
)
model.load_state_dict(torch.load("aot_style_classifier.pth"))
model.eval()

# Preprocessing
CLASS_NAMES = ["AOT-Style", "Other-Anime", "Not-Anime"]
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Bot handler
async def classify_image(update: Update, context):
    try:
        file = await update.message.photo[-1].get_file()
        await file.download_to_drive("user_image.jpg")
        
        img = Image.open("user_image.jpg").convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            pred = CLASS_NAMES[torch.argmax(output).item()]
            conf = torch.softmax(output, dim=1)[0].max().item()
        
        await update.message.reply_text(
            f"🎨 Predicted: {pred}\n"
            f"🔍 Confidence: {conf:.0%}"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# Start bot
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # Set in Render dashboard
app = Application.builder().token(TELEGRAM_TOKEN).build()
app.add_handler(MessageHandler(filters.PHOTO, classify_image))
app.run_polling()