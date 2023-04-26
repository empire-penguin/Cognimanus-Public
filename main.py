import torch
import numpy as np
from tqdm import tqdm

NUM_EPOCHS = 1000

# Create a torch.nn module that will classify a binary AND gate
class AND(torch.nn.Module):
    def __init__(self):
        super(AND, self).__init__()
        self.fc1 = torch.nn.Linear(2, 2)
        self.fc2 = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class Hand_Position_Model(torch.nn.module):
    def __init__(self) -> None:
        super(Hand_Position_Model).__init__():
        

def main():
    model = AND()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in tqdm(range(NUM_EPOCHS)):
        # Forward pass
        x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
        y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float)
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}')

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test the model
    x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
    y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float)
    y_pred = model(x)

    # Print the model's predictions
    print('After training:')
    for i in range(4):
        print(f'Input: {x[i].numpy()} | Ground Truth: {y[i].numpy()} | Prediction: {y_pred[i].item():.4f}')


if __name__ == '__main__':
    main()
