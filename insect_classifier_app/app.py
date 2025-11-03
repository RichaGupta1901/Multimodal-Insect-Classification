from flask import Flask, render_template, request
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms
from PIL import Image
import timm
import numpy as np

app = Flask(__name__)

# -------------------------
# Encoders (same as training)
# -------------------------
class ImageEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, proj_dim=256, feat_pool='avg'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool=feat_pool)
        feat_dim = self.backbone.num_features
        self.feat_dim = feat_dim
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, proj_dim)
        )

    def forward(self, x):
        feat = self.backbone(x)
        z = self.proj(feat)
        return feat, z


class DNAEncoder(nn.Module):
    def __init__(self, hf_model_name='zhihan1996/DNA_bert_6', proj_dim=256, freeze_backbone=False):
        super().__init__()
        self.model = AutoModel.from_pretrained(hf_model_name)
        hidden = self.model.config.hidden_size
        self.feat_dim = hidden
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, proj_dim)
        )
        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0, :]
        z = self.proj(pooled)
        return pooled, z


class EnvEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[256,128], proj_dim=256, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.feat_dim = prev
        self.proj = nn.Linear(prev, proj_dim)

    def forward(self, x):
        h = self.mlp(x)
        z = self.proj(h)
        return h, z


# -------------------------
# Multimodal Model
# -------------------------
class MultiModalNet(nn.Module):
    def __init__(self, num_classes=5, proj_dim=256):
        super().__init__()
        self.img_enc = ImageEncoder(proj_dim=proj_dim)
        self.dna_enc = DNAEncoder(proj_dim=proj_dim)
        self.env_enc = EnvEncoder(proj_dim=proj_dim)

        self.proj_dim = proj_dim
        self.missing_token = nn.Parameter(torch.randn(proj_dim) * 0.02)

        cls_in = proj_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, img=None, dna_input_ids=None, dna_attention_mask=None, env=None):
        device = next(self.parameters()).device
        mt = self.missing_token.view(1, -1).to(device)

        # Image
        if img is not None:
            _, z_img = self.img_enc(img)
        else:
            z_img = mt.repeat(1, 1)

        # DNA
        if dna_input_ids is not None and dna_attention_mask is not None:
            _, z_dna = self.dna_enc(dna_input_ids, dna_attention_mask)
        else:
            z_dna = mt.repeat(1, 1)

        # Env
        if env is not None:
            _, z_env = self.env_enc(env)
        else:
            z_env = mt.repeat(1, 1)

        cls_input = torch.cat([z_img, z_dna, z_env], dim=1)
        logits = self.classifier(cls_input)
        return logits


# -------------------------
# Load model + tokenizer
# -------------------------
device = torch.device("cpu")
model = MultiModalNet(num_classes=4, proj_dim=256)
state_dict = torch.load("E:\Symbiosis\Studies\Sem 7\MMA Project\insect_classifier_app\model\insect_model_hybrid.pth", map_location=device)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

classes = ["Diptera", "Lepidoptera", "Hymenoptera", "Coleoptera", "Hemiptera"]

# -------------------------
# Routes
# -------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    dna_seq = request.form.get('dna_seq', '').strip()
    env_text = request.form.get('env_info', '').strip()

    # Image
    img_tensor = None
    if 'file' in request.files and request.files['file'].filename != '':
        img = Image.open(request.files['file'].stream).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

    # DNA (convert to 6-mer tokens)
    dna_input_ids, dna_mask = None, None
    if dna_seq:
        tokens = tokenizer(dna_seq, return_tensors='pt', truncation=True, padding=True, max_length=512)
        dna_input_ids = tokens['input_ids'].to(device)
        dna_mask = tokens['attention_mask'].to(device)

    # Environment info (simple text-to-numeric mockup)
    env_tensor = None
    if env_text:
        # Convert text to 3 dummy features for demo (can replace with real env embeddings)
        env_tensor = torch.tensor(np.random.rand(1, 3), dtype=torch.float32).to(device)

    # Predict
    with torch.no_grad():
        logits = model(img=img_tensor, dna_input_ids=dna_input_ids, dna_attention_mask=dna_mask, env=env_tensor)
        pred = torch.argmax(logits, dim=1).item()
        label = classes[pred]

    return render_template('result.html', label=label)


if __name__ == '__main__':
    app.run(debug=True)
