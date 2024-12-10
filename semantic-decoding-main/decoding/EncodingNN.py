class EncodingNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, bottleneck_size=32):
        super(EncodingNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.bottleneck = nn.Linear(hidden_size, bottleneck_size)
        self.fc2 = nn.Linear(bottleneck_size, output_size)
        self.noise_fc = nn.Linear(output_size, output_size)

    def forward(self, x, y=None):  # Accept `targets` as an optional argument
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bottleneck(x)
        preds = self.fc2(x)

        if y is not None:  # Compute residuals if `targets` are provided
            residuals = y - preds
        else:
            residuals = torch.zeros_like(preds)  # Placeholder if no targets

        noise_structure = self.noise_fc(residuals)
        return preds, noise_structure