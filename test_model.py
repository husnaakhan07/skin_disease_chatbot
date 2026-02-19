import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MODEL_PATH = 'skin_disease_model.pth'
DATA_DIR = 'data/test_data'
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the "Brain"
checkpoint = torch.load(MODEL_PATH)
class_names = checkpoint['class_names']
num_classes = len(class_names)

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

# 2. Prepare Test Data
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(DATA_DIR, test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Evaluation Loop
all_preds = []
all_labels = []

print("🚀 Running final evaluation on test_data...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 4. Results
print("\n--- 📊 Performance Report ---")
# This handles the case where some classes might be missing in test_data
target_names = [class_names[i] for i in sorted(list(set(all_labels)))]
print(classification_report(all_labels, all_preds, target_names=target_names))

# 5. Visualize Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(15, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Skin Disease Diagnostic Confusion Matrix')
plt.show()