import sys
import time
import subprocess
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

@st.cache_resource
def load_model(model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # Set the output according to the number of existing class
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model
