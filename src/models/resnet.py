class Resnet:
    def __init__(self, input_dim, output_dim, hidden_dim, n_blocks, dr):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dr = dr

    def forward(self, x):
        return x
