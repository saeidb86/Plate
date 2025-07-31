import torch
import torch.nn as nn
from torchvision import transforms

from core.model import CRNN

class PlateRecognizer:
    def __init__(self, model_path, char_set):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.char_set = char_set
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def _load_model(self, model_path):
        model = CRNN(num_classes=len(self.char_set)+1)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def recognize(self, plate_image):
        image_tensor = self.transform(plate_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.model(image_tensor)
            _, max_indices = torch.max(preds, 2)
            text = self._decode_prediction(max_indices[0])
        
        return text
    
    def _decode_prediction(self, pred):
        decoded = []
        prev_char = None
        
        for char_idx in pred.cpu().numpy():
            if char_idx != 0 and char_idx != prev_char:
                decoded.append(char_idx)
            prev_char = char_idx if char_idx != 0 else None
        
        return ''.join([self.char_set[idx-1] for idx in decoded if 0 < idx <= len(self.char_set)])