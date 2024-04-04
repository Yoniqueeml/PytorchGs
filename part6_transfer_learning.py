import pandas as pd
import torch.cuda
from PIL import Image
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from torch import nn
from torchmetrics import ConfusionMatrix
from pathlib import Path

from helper_functions import plot_loss_curves
from part5_modular import data_setup, engine
from torchvision import transforms
import torchvision
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_path = 'data/pizza_steak_sushi'
train_dir = image_path + "/train"
test_dir = image_path + "/test"

simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=simple_transform,
                                                                               batch_size=32)
model_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model_0 = torchvision.models.efficientnet_b0(weights=model_weights).to(device)
for param in model_0.features.parameters():
    param.requires_grad = False
output_shape = len(class_names)
model_0.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=output_shape,
                    bias=True)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_0.parameters(), lr=0.001)

model_0_results = engine.train(model=model_0,
                               train_dataloader=train_dataloader,
                               test_dataloader=test_dataloader,
                               optimizer=optimizer,
                               loss_fn=loss_fn,
                               epochs=5,
                               device=device)

plot_loss_curves(model_0_results)
test_preds = []
model_0.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader):
        X, y = X.to(device), y.to(device)
        test_logits = model_0(X)

        pred_probs = torch.softmax(test_logits, dim=1)

        pred_labels = torch.argmax(pred_probs, dim=1)

        test_preds.append(pred_labels)

test_preds = torch.cat(test_preds).cpu()
test_truth = torch.cat([y for X, y in test_dataloader])
confmat = ConfusionMatrix(task='multiclass', num_classes=len(class_names))
confmat_tensor = confmat(preds=test_preds,
                         target=test_truth)

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(10, 7)
)
plt.show()

test_data_paths = list(Path(test_dir).glob("*/*.jpg"))
test_labels = [path.parent.stem for path in test_data_paths]


def pred_and_store(test_paths, model, transform, class_names, device):
    test_pred_list = []
    for path in tqdm(test_paths):
        pred_dict = {}
        pred_dict["image_path"] = path
        class_name = path.parent.stem
        pred_dict["class_name"] = class_name
        img = Image.open(path)
        transformed_image = transform(img).unsqueeze(0)
        model.eval()
        with torch.inference_mode():
            pred_logit = model(transformed_image.to(device))
            pred_prob = torch.softmax(pred_logit, dim=1)
            pred_label = torch.argmax(pred_prob, dim=1)
            pred_class = class_names[pred_label.cpu()]

            pred_dict["pred_prob"] = pred_prob.unsqueeze(0).max().cpu().item()
            pred_dict["pred_class"] = pred_class

        pred_dict["correct"] = class_name == pred_class

        test_pred_list.append(pred_dict)

    return test_pred_list


test_pred_dicts = pred_and_store(test_paths=test_data_paths,
                                 model=model_0,
                                 transform=simple_transform,
                                 class_names=class_names,
                                 device=device)
test_pred_df = pd.DataFrame(test_pred_dicts)
top_5_most_wrong = test_pred_df.sort_values(by=["correct", "pred_prob"], ascending=[True, False]).head()
print(top_5_most_wrong.head())

for row in top_5_most_wrong.iterrows():
    row = row[1]
    image_path = row[0]
    true_label = row[1]
    pred_prob = row[2]
    pred_class = row[3]
    img = torchvision.io.read_image(str(image_path))
    plt.figure()
    plt.imshow(img.permute(1, 2, 0))
    plt.title(f"True: {true_label} | Pred: {pred_class} | Prob: {pred_prob:.3f}")
    plt.axis(False)
