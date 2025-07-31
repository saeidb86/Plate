import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(CRNN, self).__init__()
        
        # CNN Layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        
        # RNN Layers
        self.rnn = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # CNN
        conv = self.cnn(x)
        batch_size, channels, height, width = conv.size()
        
        # Prepare for RNN
        conv = conv.squeeze(2)  # Remove height dimension
        conv = conv.permute(0, 2, 1)  # (batch, width, channels)
        
        # RNN
        rnn_out, _ = self.rnn(conv)
        
        # Output
        output = self.fc(rnn_out)
        return output

if __name__ == "__main__":
    # Test the model
    model = CRNN(num_classes=43)
    test_input = torch.randn(1, 1, 32, 128)
    test_output = model(test_input)
    print("Test output shape:", test_output.shape)