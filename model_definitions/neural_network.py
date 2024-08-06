import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        weights = torch.softmax(self.attention_weights(x), dim=0)
        return x * weights


class SimpleNNWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNNWithAttention, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.attention = Attention(hidden_size)  # Attention mechanism
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return torch.sigmoid(x)

def train_neural_network(X_train, y_train, input_size, hidden_size, output_size, num_epochs=710, learning_rate=0.01):
    # Convert DataFrame to numpy array and then to tensor
    try:
        X_train_np = X_train.to_numpy(dtype=np.float32)
        print(f"X_train converted to numpy array with shape: {X_train_np.shape}")
    except Exception as e:
        print(f"Error converting X_train to numpy array: {e}")
        raise e

    try:
        X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
        print(f"X_train_tensor shape: {X_train_tensor.shape}")
    except Exception as e:
        print(f"Error converting X_train to tensor: {e}")
        raise e

    try:
        y_train_np = y_train.to_numpy(dtype=np.float32)
        print(f"y_train converted to numpy array with shape: {y_train_np.shape}")
    except Exception as e:
        print(f"Error converting y_train to numpy array: {e}")
        raise e

    try:
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
        print(f"y_train_tensor shape: {y_train_tensor.shape}")
    except Exception as e:
        print(f"Error converting y_train to tensor: {e}")
        raise e

    model = SimpleNNWithAttention(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model


def evaluate_neural_network(model, X_test, y_test):
    try:
        X_test_np = X_test.to_numpy(dtype=np.float32)
        print(f"X_test converted to numpy array with shape: {X_test_np.shape}")
    except Exception as e:
        print(f"Error converting X_test to numpy array: {e}")
        raise e

    try:
        X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
        print(f"X_test_tensor shape: {X_test_tensor.shape}")
    except Exception as e:
        print(f"Error converting X_test to tensor: {e}")
        raise e

    try:
        y_test_np = y_test.to_numpy(dtype=np.float32)
        print(f"y_test converted to numpy array with shape: {y_test_np.shape}")
    except Exception as e:
        print(f"Error converting y_test to numpy array: {e}")
        raise e

    try:
        y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32).view(-1, 1)
        print(f"y_test_tensor shape: {y_test_tensor.shape}")
    except Exception as e:
        print(f"Error converting y_test to tensor: {e}")
        raise e

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predicted = (outputs >= 0.5).float()
        accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
        print(f'Accuracy of the network on the test data: {accuracy * 100:.2f}%')

        # Convert predictions and labels to numpy arrays for scikit-learn functions
        predicted_np = predicted.numpy()
        y_test_np = y_test_tensor.numpy()

        # Calculate classification report and confusion matrix
        report = classification_report(y_test_np, predicted_np, zero_division=0)
        conf_matrix = confusion_matrix(y_test_np, predicted_np)
        print(f"Classification Report for neural_network:\n{report}")
        print(f"Confusion Matrix for neural_network:\n{conf_matrix}")

    return accuracy, report, conf_matrix
