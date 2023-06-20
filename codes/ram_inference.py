import torch
import torchvision.transforms as transforms
from RAM.models.tag2text import ram, tag2text_caption
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# init image transforms
image_size = 384
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# load RAM Model
model_ram = ram(
    pretrained='./pretrained/ram_swin_large_14m.pth',
    image_size=image_size,
    vit='swin_l'
).eval().to(device)

def inference_with_ram(img_path):
    with torch.no_grad():
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        tags, tags_chinese = model_ram.generate_tag(img)
        return tags[0]

result = inference_with_ram('./data/bird3.png')
print(result)


