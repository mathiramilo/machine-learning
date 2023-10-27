from torch import nn


class FeedForwardNN(nn.Module):
    def __init__(
        self, hl_amount=1, hl_size=32, f_activation=nn.Sigmoid(), dropout_prob=0.5
    ):
        super(FeedForwardNN, self).__init__()
        self.input_layer = nn.Linear(28 * 28, hl_size)
        self.activation = f_activation
        self.dropout = nn.Dropout(
            dropout_prob
        )  # Capa Dropout con la tasa de Dropout especificada
        for i in range(hl_amount - 1):
            setattr(self, f"hidden_layer_{i}", nn.Linear(hl_size, hl_size))
        self.output_layer = nn.Linear(hl_size, 10)
        self.hl_amount = hl_amount

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.input_layer(x)
        x = self.activation(x)
        for i in range(self.hl_amount - 1):
            x = getattr(self, f"hidden_layer_{i}")(x)
            x = self.activation(x)
            x = self.dropout(x)  # Aplicar Dropout después de cada capa oculta
        x = self.output_layer(x)
        return x
