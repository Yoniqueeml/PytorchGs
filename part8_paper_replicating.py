import torch
import torchvision
from torch import nn
from torchinfo import summary
from torchvision import transforms
from part5_modular import data_setup, engine
from helper_functions import download_data, set_seeds, plot_loss_curves

device = "cuda" if torch.cuda.is_available() else "cpu"
image_path = download_data(
    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    destination="pizza_steak_sushi")

train_dir = image_path / "train"
test_dir = image_path / "test"

IMG_SIZE = 224

manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

BATCH_SIZE = 32

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transforms,
    batch_size=BATCH_SIZE
)


class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """

    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 embedding_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        self.flatten = nn.Flatten(start_dim=2,
                                  end_dim=3)

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0, 2, 1)


class ViT(nn.Module):
    def __init__(self,
                 img_size=224,  # from Table
                 num_channels=3,
                 patch_size=16,
                 embedding_dim=768,  # from Table
                 dropout=0.1,
                 mlp_size=3072,  # from Table
                 num_transformer_layers=12,  # from Table
                 num_heads=12,  # from Table  (number of multi-head self attention heads)
                 num_classes=1000):  # generic number of classes
        super().__init__()

        assert img_size % patch_size == 0, "Image size must be divisble by patch size."

        self.patch_embedding = PatchEmbedding(in_channels=num_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                        requires_grad=True)

        num_patches = (img_size * img_size) // patch_size ** 2  # N = HW/P^2
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))

        self.embedding_dropout = nn.Dropout(p=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                                                                  nhead=num_heads,
                                                                                                  dim_feedforward=mlp_size,
                                                                                                  activation="gelu",
                                                                                                  batch_first=True,
                                                                                                  norm_first=True),
                                                         num_layers=num_transformer_layers)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x)
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.positional_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.mlp_head(x[:, 0])
        return x


demo_img = torch.randn(1, 3, 224, 224).to(device)
summary(model=ViT(num_classes=3),
        input_size=demo_img.shape)

vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit = torchvision.models.vit_b_16(weights=vit_weights)
for param in pretrained_vit.parameters():
    param.requires_grad = False
embedding_dim = 768
set_seeds()
pretrained_vit.heads = nn.Sequential(
    nn.LayerNorm(normalized_shape=embedding_dim),
    nn.Linear(in_features=embedding_dim,
              out_features=len(class_names))
)
summary(model=pretrained_vit,
        input_size=(1, 3, 224, 224),  # (batch_size, color_channels, height, width)
        col_names=["input_size"],
        col_width=20,
        row_settings=["var_names"]
        )

optimizer = torch.optim.Adam(params=pretrained_vit.parameters(),
                             lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
pretrained_vit_results = engine.train(model=pretrained_vit,
                                           train_dataloader=train_dataloader,
                                           test_dataloader=test_dataloader,
                                           optimizer=optimizer,
                                           loss_fn=loss_fn,
                                           epochs=10,
                                           device=device)
plot_loss_curves(pretrained_vit_results)
