from src.models.lstm import SimpleLSTM
from src.utils.load_data import stock_dataloader
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn

def round():
    # Hyperparameters
    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    num_epochs = 10
    learning_rate = 0.001
    ticker = "AAPL"

    train_loader, test_loader = stock_dataloader(ticker, "2020-01-01", "2023-01-01")

    model = SimpleLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    loss_function = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            
            # Make predictions
            outputs = model(inputs)
            
            # Compute loss
            loss = loss_function(outputs, targets.view(-1, 1))
            
            # Backpropagate and optimize
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_predictions = []
        test_targets = []
        for inputs, targets in test_loader:
            outputs = model(inputs)
            test_predictions.append(outputs.detach().numpy().tolist())
            test_targets.append(targets.numpy().tolist())

    print("Evaluation completed!")  

    test_predictions_flattened = [item for sublist in test_predictions for item in sublist]
    test_targets_flattened = [item for sublist in test_targets for item in sublist]

    # Plot real vs predicted stock prices
    plt.figure(figsize=(14,6))
    plt.plot(test_targets_flattened, label='True Prices', color='blue')
    plt.plot(test_predictions_flattened, label='Predicted Prices', color='red', alpha=0.6)
    plt.title('Stock Price Predictions')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./src/img/LSTM_{ticker}.png")



if __name__ == "__main__":
    round()