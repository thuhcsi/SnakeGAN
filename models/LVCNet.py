import logging
import math

import numpy as np
import torch

import torch.nn.functional as F


class Conv1d(torch.nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class Conv1d1x1(Conv1d):
    """1x1 Conv1d with customized initialization."""

    def __init__(self, in_channels, out_channels, bias):
        """Initialize 1x1 Conv1d module."""
        super(Conv1d1x1, self).__init__(in_channels, out_channels,
                                        kernel_size=1, padding=0,
                                        dilation=1, bias=bias)

class KernelPredictor(torch.nn.Module):
    ''' Kernel predictor for the location-variable convolutions
    '''

    def __init__(self, 
                 cond_channels,
                 conv_in_channels,
                 conv_out_channels,
                 conv_layers,
                 conv_kernel_size=3,
                 kpnet_hidden_channels=64,
                 kpnet_conv_size=1,
                 kpnet_dropout=0.0,
                 kpnet_nonlinear_activation="LeakyReLU",
                 kpnet_nonlinear_activation_params={"negative_slope":0.1}
                 ):
        '''
        Args:
            cond_channels (int): number of channel for the conditioning sequence,
            conv_in_channels (int): number of channel for the input sequence,
            conv_out_channels (int): number of channel for the output sequence,
            conv_layers (int):
            kpnet_
        '''
        super().__init__() 

        self.conv_in_channels = conv_in_channels 
        self.conv_out_channels = conv_out_channels 
        self.conv_kernel_size = conv_kernel_size 
        self.conv_layers = conv_layers

        kpnet_kernel_channels = conv_in_channels * conv_out_channels * conv_kernel_size * conv_layers 
        kpnet_bias_channels = conv_out_channels * conv_layers 

        padding = (kpnet_conv_size - 1)//2 
        self.input_conv = torch.nn.Sequential(
            torch.nn.Conv1d(cond_channels, kpnet_hidden_channels, 5, padding=2, bias=True), 
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
        )

        self.residual_conv = torch.nn.Sequential(
            torch.nn.Dropout(kpnet_dropout), 
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True), 
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True), 
        )

        self.kernel_conv = torch.nn.Conv1d(kpnet_hidden_channels, kpnet_kernel_channels, kpnet_conv_size, padding=padding, bias=True) 
        self.bias_conv = torch.nn.Conv1d(kpnet_hidden_channels, kpnet_bias_channels, kpnet_conv_size, padding=padding, bias=True) 
        
    def forward(self, c):
        '''
        Args:
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length) 
        Returns:
        '''
        batch, cond_channels, cond_length = c.shape 
        c = self.input_conv( c )
        
        c = c + self.residual_conv( c  ) 
        k = self.kernel_conv( c ) 
        b = self.bias_conv( c ) 

        kernels = k.contiguous().view(batch, 
                                      self.conv_layers,
                                      self.conv_in_channels,
                                      self.conv_out_channels,
                                      self.conv_kernel_size,
                                      cond_length) 
        bias = b.contiguous().view( batch, 
                                    self.conv_layers, 
                                    self.conv_out_channels,
                                    cond_length) 
        # print('kernels: ', kernels.shape)
        return kernels, bias



class LVCBlock(torch.nn.Module):
    ''' the location-variable convolutions 
    '''

    def __init__(self,
                 in_channels,
                 cond_channels,
                 conv_layers=10,
                 conv_kernel_size=3,
                 cond_hop_length=256,
                 kpnet_hidden_channels=64,
                 kpnet_conv_size=1,
                 kpnet_dropout=0.0
                 ):
        super().__init__() 

        self.cond_hop_length = cond_hop_length 
        self.conv_layers = conv_layers 
        self.conv_kernel_size = conv_kernel_size 

        self.kernel_predictor = KernelPredictor( 
                    cond_channels=cond_channels,
                    conv_in_channels=in_channels,
                    conv_out_channels=2*in_channels, 
                    conv_layers=conv_layers,
                    conv_kernel_size=conv_kernel_size,
                    kpnet_hidden_channels=kpnet_hidden_channels,
                    kpnet_conv_size=kpnet_conv_size,
                    kpnet_dropout=kpnet_dropout 
                    ) 

    def forward(self, x, c):
        ''' forward propagation of the location-variable convolutions.  
        Args: 
            x (Tensor): the input sequence (batch, in_channels, in_length) 
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)
        
        Returns:
            Tensor: the output sequence (batch, in_channels, in_length) 
        ''' 
        batch, in_channels, in_length = x.shape 
        batch, cond_channels, cond_length = c.shape 

        assert in_length == (cond_length * self.cond_hop_length ), ( 
            f"the length of input ({in_length}, {cond_length}) is not match in LVCNet" ) 

        kernels, bias = self.kernel_predictor( c ) 

        for i in range(self.conv_layers):
            dilation = 2**i 
            k = kernels[ :, i, :, :, :, : ] 
            b = bias[ :, i, :, : ] 
            x = self.location_variable_convolution( x, k, b, dilation, self.cond_hop_length ) 
            x = torch.sigmoid( x[ :, :in_channels, : ] ) * torch.tanh( x[ :, in_channels:, : ] ) 
        return x 
            
    
    def location_variable_convolution(self, x, kernel, bias, dilation, hop_size):
        ''' perform location-variable convolution operation on the input sequence (x) using the local convolution kernl. 
        Time: 414 μs ± 309 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each), test on NVIDIA V100. 
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length). 
            kernel (Tensor): the local convolution kernel (batch, in_channel, out_channels, kernel_size, kernel_length) 
            bias (Tensor): the bias for the local convolution (batch, out_channels, kernel_length) 
            dilation (int): the dilation of convolution. 
            hop_size (int): the hop_size of the conditioning sequence. 
        Returns:
            (Tensor): the output sequence after performing local convolution. (batch, out_channels, in_length).
        '''
        batch, in_channels, in_length = x.shape 
        batch, in_channels, out_channels, kernel_size, kernel_length = kernel.shape 

        assert in_length == (kernel_length*hop_size), "length of (x, kernel) is not matched" 

        padding = dilation * int( (kernel_size - 1) / 2 ) 
        x = F.pad( x, (padding, padding), 'constant', 0 )      # (batch, in_channels, in_length + 2*padding)
        x = x.unfold( 2, hop_size + 2 * padding, hop_size )    # (batch, in_channels, kernel_length, hop_size + 2*padding)

        if hop_size < dilation:
            x = F.pad( x, (0, dilation), 'constant', 0 ) 
        x = x.unfold(3, dilation, dilation)     # (batch, in_channels, kernel_length, (hop_size + 2*padding)/dilation, dilation)
        x = x[ :, :, :, :, :hop_size ]          
        x = x.transpose( 3, 4 )                 # (batch, in_channels, kernel_length, dilation, (hop_size + 2*padding)/dilation)  
        x = x.unfold( 4, kernel_size, 1 )       # (batch, in_channels, kernel_length, dilation, _, kernel_size)

        o = torch.einsum( 'bildsk,biokl->bolsd', x, kernel ) 
        o = o + bias.unsqueeze(-1).unsqueeze(-1) 
        o = o.contiguous().view(batch, out_channels, -1) 
        return o 



class LVCNetGenerator(torch.nn.Module):
    """Parallel WaveGAN Generator module."""

    def __init__(self,
                 in_channels=101,
                 out_channels=1,
                 inner_channels=128,
                 cond_channels=81,
                 cond_hop_length=256,
                 lvc_block_nums=2,
                 lvc_layers_each_block=5,
                 lvc_kernel_size=3,
                 kpnet_hidden_channels=64,
                 kpnet_conv_size=1,
                 dropout=0.0,
                 use_weight_norm=True,
                 ):
        """Initialize Parallel WaveGAN Generator module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of dilated convolution.
            layers (int): Number of residual block layers.
            stacks (int): Number of stacks i.e., dilation cycles.
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for auxiliary feature conv.
            aux_context_window (int): Context window size for auxiliary feature.
            dropout (float): Dropout rate. 0.0 means no dropout applied.
            bias (bool): Whether to use bias parameter in conv layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal structure.
            upsample_conditional_features (bool): Whether to use upsampling network.
            upsample_net (str): Upsampling network architecture.
            upsample_params (dict): Upsampling network parameters.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.lvc_block_nums = lvc_block_nums

        # define first convolution
        self.first_conv = Conv1d1x1(in_channels, inner_channels, bias=True)

        # define residual blocks
        self.lvc_blocks = torch.nn.ModuleList()
        for n in range(lvc_block_nums):
            lvcb = LVCBlock(
                in_channels=inner_channels, 
                cond_channels=cond_channels,
                conv_layers=lvc_layers_each_block,
                conv_kernel_size=lvc_kernel_size,
                cond_hop_length=cond_hop_length,
                kpnet_hidden_channels=kpnet_hidden_channels,
                kpnet_conv_size=kpnet_conv_size,
                kpnet_dropout=dropout,
            )
            self.lvc_blocks += [lvcb]

        # define output layers
        self.last_conv_layers = torch.nn.ModuleList([
            torch.nn.ReLU(inplace=True),
            Conv1d1x1(inner_channels, inner_channels, bias=True),
            torch.nn.ReLU(inplace=True),
            Conv1d1x1(inner_channels, out_channels, bias=True),
        ])

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, c):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T').
        Returns:
            Tensor: Output tensor (B, out_channels, T)
        """

        x = self.first_conv(x)
        x = self.lvc_blocks[0]( x, c )
        for n in range(1, self.lvc_block_nums):
            x = x + self.lvc_blocks[n]( x, c )

        # apply final layers
        for f in self.last_conv_layers:
            x = f(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    @staticmethod
    def _get_receptive_field_size(layers, stacks, kernel_size,
                                  dilation=lambda x: 2 ** x):
        assert layers % stacks == 0
        layers_per_cycle = layers // stacks
        dilations = [dilation(i % layers_per_cycle) for i in range(layers)]
        return (kernel_size - 1) * sum(dilations) + 1

    @property
    def receptive_field_size(self):
        """Return receptive field size."""
        return self._get_receptive_field_size(self.layers, self.stacks, self.kernel_size)

    def inference(self, c=None, x=None):
        """Perform inference.
        Args:
            c (Union[Tensor, ndarray]): Local conditioning auxiliary features (T' ,C).
            x (Union[Tensor, ndarray]): Input noise signal (T, 1).
        Returns:
            Tensor: Output tensor (T, out_channels)
        """
        if x is not None:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float).to(next(self.parameters()).device)
            x = x.transpose(1, 0).unsqueeze(0)
        else:
            assert c is not None
            x = torch.randn(1, 1, len(c) * self.upsample_factor).to(next(self.parameters()).device)
        if c is not None:
            if not isinstance(c, torch.Tensor):
                c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
            c = c.transpose(1, 0).unsqueeze(0)
            c = torch.nn.ReplicationPad1d(self.aux_context_window)(c)
        return self.forward(x, c).squeeze(0).transpose(1, 0)