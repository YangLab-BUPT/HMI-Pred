import torch
import torch.nn as nn

class HMIS(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
    
        super(HMIS, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512,64)
        #self.fc3 = nn.Linear(256, 64)
        #self.fc5 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, output_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  

    
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)  
        #x = self.fc3(x)
        #x = self.relu(x)
        #x = self.dropout(x)  
        #x = self.fc5(x)
        x = self.fc4(x)
        return x  
        
        
