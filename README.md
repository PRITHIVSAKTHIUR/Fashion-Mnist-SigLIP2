![zsfgzdsfrg.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/uE2UMwmSZYc8KCE06QtTi.png)

# **Fashion-Mnist-SigLIP2**
> **Fashion-Mnist-SigLIP2** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to classify images into **Fashion-MNIST** categories using the **SiglipForImageClassification** architecture.  

```py
Classification Report:
               precision    recall  f1-score   support

T-shirt / top     0.8142    0.9147    0.8615      6000
      Trouser     0.9935    0.9870    0.9902      6000
     Pullover     0.8901    0.8610    0.8753      6000
        Dress     0.9098    0.9300    0.9198      6000
         Coat     0.8636    0.8865    0.8749      6000
       Sandal     0.9857    0.9847    0.9852      6000
        Shirt     0.8076    0.6962    0.7478      6000
      Sneaker     0.9663    0.9695    0.9679      6000
          Bag     0.9779    0.9805    0.9792      6000
   Ankle boot     0.9698    0.9700    0.9699      6000

     accuracy                         0.9180     60000
    macro avg     0.9179    0.9180    0.9172     60000
 weighted avg     0.9179    0.9180    0.9172     60000
```

![Untitled.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/4RcQ0vyPssALOOCIhpNqu.png)

The model categorizes images into the following 10 classes:  
- **Class 0:** "T-shirt / top"  
- **Class 1:** "Trouser"  
- **Class 2:** "Pullover"  
- **Class 3:** "Dress"  
- **Class 4:** "Coat"  
- **Class 5:** "Sandal"  
- **Class 6:** "Shirt"  
- **Class 7:** "Sneaker"  
- **Class 8:** "Bag"  
- **Class 9:** "Ankle boot"  

# **Run with Transformers🤗**

```python
!pip install -q transformers torch pillow gradio
```

```python
import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Fashion-Mnist-SigLIP2"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def fashion_mnist_classification(image):
    """Predicts fashion category for an image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "T-shirt / top", "1": "Trouser", "2": "Pullover", "3": "Dress", "4": "Coat",
        "5": "Sandal", "6": "Shirt", "7": "Sneaker", "8": "Bag", "9": "Ankle boot"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=fashion_mnist_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Fashion MNIST Classification",
    description="Upload an image to classify it into one of the 10 Fashion-MNIST categories."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
```

# **Intended Use:**  

The **Fashion-Mnist-SigLIP2** model is designed for fashion image classification. It helps categorize clothing and footwear items into predefined Fashion-MNIST classes. Potential use cases include:  

- **Fashion Recognition:** Classifying fashion images into common categories like shirts, sneakers, and dresses.  
- **E-commerce Applications:** Assisting online retailers in organizing and tagging clothing items for better search and recommendations.  
- **Automated Fashion Sorting:** Helping automated inventory management systems classify fashion items.  
- **Educational Purposes:** Supporting AI and ML research in vision-based fashion classification models.
