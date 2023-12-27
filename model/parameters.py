# coding = utf-8
class Parameters:

    def __init__(self, d_model=512, d_ff=2048, d_k=64, d_v=64, n_layers=8, n_heads=8):
        self._d_model = d_model  # Embedding Size
        self._d_ff = d_ff  # FeedForward Dimension
        self._d_k = d_k  # Dimension of K(=Q)
        self._d_v = d_v  # Dimension of V
        self._n_layers = n_layers  # Number of Encoder/Decoder Layers
        self._n_heads = n_heads  # Number of heads in Multi-Head Attention

    @property
    def d_model(self):
        return self._d_model

    @property
    def d_ff(self):
        return self._d_ff

    @property
    def d_k(self):
        return self._d_k

    @property
    def d_v(self):
        return self._d_v

    @property
    def n_layers(self):
        return self._n_layers

    @property
    def n_heads(self):
        return self._n_heads


p = Parameters()
