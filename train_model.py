import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
from PIL import Image

# --- CONFIGURATION ---
DATA_DIR = 'data' 
BATCH_SIZE = 4
EPOCHS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Now 'transforms' is defined and safe to use!
data_transforms = {
    'train_data': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test_data': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {}
for x in ['train_data', 'test_data']:
    path = os.path.join(DATA_DIR, x)
    
    if not os.path.exists(path):
        print(f"❌ CRITICAL: Folder not found at {os.path.abspath(path)}")
        continue

    print(f"--- Checking {x} ---")
    # Load into PyTorch
    try:
        # We use the specific transform defined above for each key
        image_datasets[x] = datasets.ImageFolder(path, data_transforms[x])
        print(f"✅ Success: Loaded {len(image_datasets[x])} images for {x}.\n")
    except Exception as e:
        print(f"❌ FAILED to load {x}: {e}")

# Safety Exit - Ensure BOTH are loaded
if 'train_data' not in image_datasets or 'test_data' not in image_datasets:
    print("FATAL: Either train_data or test_data failed to load. Check errors above.")
    exit()

# Now this will not throw a KeyError because 'test_data' exists in image_datasets
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True)
              for x in ['train_data', 'test_data']}

class_names = image_datasets['train_data'].classes
num_classes = len(class_names)

# --- 2. MODEL SETUP ---
model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# --- 3. TRAINING LOOP ---
print(f"Starting Training on {DEVICE} for {num_classes} classes...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    # Wrap the dataloader with tqdm for a progress bar
    # 'desc' shows the current epoch in the bar
    pbar = tqdm(dataloaders['train_data'], desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # Update the progress bar with the current loss
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    epoch_loss = running_loss / len(image_datasets['train_data'])
    print(f"✅ Epoch {epoch+1} Complete. Average Loss: {epoch_loss:.4f}")

# --- 4. SAVE THE BRAIN ---
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names
}, 'skin_disease_model.pth')

print("Success: Model saved as skin_disease_model.pth")