import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
            

class MotionPotential(nn.Module):
    def __init__(self, input_ch_xyz=3, input_ch_feature=4, output_ch=3, skips=[4], no_bias=False):
        """ 
        """
        super(MotionPotential, self).__init__()
        self.input_ch_xyz = input_ch_xyz
        self.input_ch_feature = input_ch_feature
        
        self.motion_mlp = nn.ModuleList(
            [DenseLayer(input_ch_xyz + input_ch_feature, 512, activation="relu")] + \
            [DenseLayer(512, 128, activation="relu")] 
            )
        
        if not no_bias:
          self.output_linear = DenseLayer(128, output_ch, activation="linear")
        else:
          self.output_linear = DenseLayer(128, output_ch, activation="linear", bias=False)

    def initialize_weights(self):
        """
        Custom method to initialize weights and biases of the model to achieve (1, 0, 0, 0) output.
        """
        for layer in self.motion_mlp:
            if isinstance(layer, DenseLayerQuat):
                # Use Xavier initialization for intermediate layers
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                # Ensure biases are set to zero
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Set the output layer weights and biases
        nn.init.zeros_(self.output_linear.weight)

        if self.output_linear.bias is not None:
            nn.init.zeros_(self.output_linear.bias)

    def forward(self, x):
        h = x

        for i, l in enumerate(self.motion_mlp):
            h = self.motion_mlp[i](h)
            h = F.relu(h)
            
        potential = self.output_linear(h)

        return potential

class DenseLayerQuat(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        """
        Initialize weights and biases of the DenseLayer.
        """
        # Use Xavier initialization for training, but we'll set weights to zero for testing purposes
        if self.activation != "linear":
            torch.nn.init.xavier_uniform_(self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        else:
            # For the output layer (linear), we'll set weights to zero to ensure zero output initially
            torch.nn.init.zeros_(self.weight)

        # Set biases to zero
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class MotionQuat(nn.Module):
    def __init__(self, input_ch_xyz=30, input_ch_feature=384, output_ch=4, skips=[4], no_bias=False):
        """Initialize a MotionQuat model with specific configurations."""
        super(MotionQuat, self).__init__()
        self.input_ch_xyz = input_ch_xyz
        self.input_ch_feature = input_ch_feature

        # Define the motion MLP as a list of dense layers
        self.motion_mlp = nn.ModuleList(
            [DenseLayerQuat(input_ch_xyz + input_ch_feature, 512, activation="relu")] +
            [DenseLayerQuat(512, 128, activation="relu")]
        )

        # Define the output layer, potentially without bias
        if not no_bias:
            self.output_linear = DenseLayerQuat(128, output_ch, activation="linear")
        else:
            self.output_linear = DenseLayerQuat(128, output_ch, activation="linear", bias=False)

        # Initialize weights and biases
        self.initialize_weights()

    def initialize_weights(self):
        """
        Custom method to initialize weights and biases of the model to achieve (1, 0, 0, 0) output.
        """
        for layer in self.motion_mlp:
            if isinstance(layer, DenseLayerQuat):
                # Use Xavier initialization for intermediate layers
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                # Ensure biases are set to zero
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Set the output layer weights and biases
        # We are specifically setting weights to zero and biases to achieve (1, 0, 0, 0)
        nn.init.zeros_(self.output_linear.weight)

        # Setting weights to identity-like behavior for quaternion output
        if self.output_linear.bias is not None:
            nn.init.zeros_(self.output_linear.bias)
            # Directly setting the first bias element to 1
            self.output_linear.bias.data[0] = 1.0

    def forward(self, x):
        """
        Perform a forward pass through the network, processing the input to generate the potential output.
        """
        # Start with input x
        h = x

        for i, l in enumerate(self.motion_mlp):
            h = l(h)  # Use the layer directly
            h = F.relu(h)  # Apply ReLU activation

        potential = self.output_linear(h)
        return potential
