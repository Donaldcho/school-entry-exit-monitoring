import torch

def load_feature_extractor():
    """Loads the feature extractor model (ResNet)."""
    feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    feature_extractor = torch.nn.Sequential(*(list(feature_extractor.children())[:-1]))
    feature_extractor.eval()
    return feature_extractor
