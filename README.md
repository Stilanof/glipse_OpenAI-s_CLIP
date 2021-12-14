# glipse_openAI_clip

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs.

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>  
Inference Notebook: <a href="https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=20></a>  

Disclaimer: the author do not own any rights for the code.

## Practice loading the model
```
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
```

### Another example
```
#let's try another model with my Linkedin profile picture! 

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50x16", device=device)

image = preprocess(Image.open("yo 4.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["preffesional", "unprofessional", "super professional"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  #prints:[[0.00197463 0.08398551 0.91403985]]
```


## Acknowledgments
Read how all works here ("https://github.com/openai/CLIP#clipavailable_models")

