import numpy as np
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, CLIPModel


class CLIPImageClassifier:
    """
    A class to encapsulate the CLIP model for image classification.
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initializes the CLIP model and processor.

        :param model_name: The name of the pretrained CLIP model to use.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the model and move it to the appropriate device (CPU/GPU)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

        # Load the processor for text and image inputs
        self.processor = AutoProcessor.from_pretrained(model_name)


        # Print model details
        self.print_model_info()

    def print_model_info(self):
        """Prints key details about the CLIP model."""
        total_params = np.sum([int(np.prod(p.shape)) for p in self.model.parameters()])
        self.input_resolution = self.model.vision_model.config.image_size
        self.context_length = self.model.text_model.config.max_position_embeddings
        self.vocab_size = self.model.text_model.config.vocab_size
        print(f"Model parameters: {total_params:,}")
        print(f"Input resolution: {self.input_resolution}")
        print(f"Context length: {self.context_length}")
        print(f"Vocab size: {self.vocab_size}")

    def classify_image(self, image_url, text_labels):
        """
        Classifies an image based on text labels.

        :param image_url: URL of the image to classify.
        :param text_labels: List of text labels for classification.
        :return: Probabilities of the image matching each label.
        """
        # Load the image from the URL
        image = Image.open(requests.get(image_url, stream=True).raw)

        # Process inputs (text labels and image)
        inputs = self.processor(text=text_labels, images=image, return_tensors="pt", padding=True).to(self.device)

        # Get model outputs
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # Image-text similarity scores
        probs = logits_per_image.softmax(dim=1)  # Convert to probabilities

        return probs


# Example usage
if __name__ == "__main__":
    # Instantiate the classifier
    classifier = CLIPImageClassifier()

    # Image URL
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # Define text labels
    labels = ["a photo of a cat", "a photo of a dog"]

    # Perform classification
    probabilities = classifier.classify_image(image_url, labels)

    # Print the probabilities
    print("Probabilities: ", probabilities)
