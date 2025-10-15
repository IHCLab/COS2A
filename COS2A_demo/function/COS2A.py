import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ResBlock
class ResBlock(nn.Module):
    def __init__(self, n_channels, kernel_size):
        super(ResBlock, self).__init__()
        assert n_channels % 4 == 0, f"n_channels ({n_channels}) must be divisible by 4 for group convolution"
        
        self.resblock = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size, stride=1, padding=kernel_size //2, bias=True, groups=4), #***#
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, n_channels, kernel_size, stride=1, padding=kernel_size //2, bias=True, groups=4), #***#
        )
        self.relu = nn.ReLU()

    def forward(self,x):
        res = self.resblock(x)
        x = res + x
        return self.relu(x)

# Model 1
class ProximalOperator(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ProximalOperator, self).__init__()
        self.prox_network = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, bias=True, groups=4), #***#
            nn.ReLU(inplace=True),
            ResBlock(channels, kernel_size),
            ResBlock(channels, kernel_size),
            ResBlock(channels, kernel_size),
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, bias=True) #***#
        )
        
    def forward(self, x):
        prox = self.prox_network(x)
        x = prox + x
        return x

# SFC Layer
class SymmetricLinear(nn.Module):
    def __init__(self, size):
        super(SymmetricLinear, self).__init__()
        self.size = size

        # Lower triangular parameters
        self.lower_triangular = nn.Parameter(torch.randn(size, size))
        self.bias_param = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        # Construct symmetric weight matrix
        W = torch.tril(self.lower_triangular) + torch.tril(self.lower_triangular, -1).T
        out = x @ W.T
        out = out + self.bias_param
        return out


class InitStage(nn.Module):
    def __init__(self, shared_D, shared_prox_operator, shared_channel_fc, in_channels=12, out_channels=172):
        super(InitStage, self).__init__()
        self.in_channels = in_channels      # MSI channels (c)
        self.out_channels = out_channels    # HSI channels (C)
        
        # Use the shared components (required parameters)
        self.D = shared_D
        self.prox_operator = shared_prox_operator
        self.channelFc = shared_channel_fc
        
        self.rho = nn.Parameter(torch.ones(1, device=device))
        
        # Direct upsampling using only 1×1 convolutions (Y_H^0)
        self.initial_estimate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # Direct 1×1 conv from MSI to HSI channels
            nn.ReLU(inplace=True)
        )
        
        self.conv1x1_D = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1)
        self.conv1x1_U = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    
    # Z^{k+1} = Prox_{1/rho DIP}(Y_h^k - U^k)
    def Z_update(self, Yh, U):
        Z = self.prox_operator(Yh - U)
        return Z

    def Yh_update(self, Z, U, D_transpose_Ys):        
        rhs = 2 * D_transpose_Ys + self.rho * (Z + U) # 2 * D^T * Ys + rho * (Z + U)

        # Model 2
        A = self.conv1x1_D(rhs)     # Downsample
        batch_size, channels, height, width = A.shape
        A_flat = A.permute(0, 2, 3, 1).reshape(-1, channels)
        Phi = self.channelFc(A_flat)
        N = Phi.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        B = self.conv1x1_U(N)       # Upsample        
        Yh = (rhs - (2 / self.rho) * B)/self.rho
        return Yh
    
    # U^{k+1} = U^k - Y_h^k + Z^{k+1}
    def U_update(self, U, Yh, Z):
        return U - Yh + Z
    
    def forward(self, Ys):
        Y_init = self.initial_estimate(Ys)      # Initialize Z
        U = torch.zeros_like(Y_init)            # Initialize U = 0        
        batch_size, _, height, width = Ys.shape

        Ys_flat = Ys.reshape(batch_size, self.in_channels, -1) # [B, C_in, H, W] -> [B, C_in, H*W]        
        # Compute D^T * Y: [C_out, C_in] * [B, C_in, H*W] -> [B, C_out, H*W]
        D_transpose_Ys = torch.matmul(self.D.transpose(0, 1).squeeze(-1).squeeze(-1), Ys_flat)
        D_transpose_Ys = D_transpose_Ys.reshape(batch_size, self.out_channels, height, width) # [B, C_out, H*W] -> [B, C_out, H, W]        
        Z = self.Z_update(Y_init, U)        
        Yh = self.Yh_update(Z, U, D_transpose_Ys)        
        U = self.U_update(U, Yh, Z)
            
        return Yh, U

class MidStage(nn.Module):
    def __init__(self, shared_D, shared_prox_operator, shared_channel_fc, in_channels=12, out_channels=172, num_iterations=1):
        super(MidStage, self).__init__()
        self.in_channels = in_channels      # MSI channels (c)
        self.out_channels = out_channels    # HSI channels (C)
        self.num_iterations = num_iterations
        
        # Use the shared components (required parameters)
        self.D = shared_D
        self.prox_operator = shared_prox_operator
        self.channelFc = shared_channel_fc
        
        self.rho = nn.Parameter(torch.ones(1, device=device))

        self.conv1x1_D = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1)
        self.conv1x1_U = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    
    # Z^{k+1} = Prox_{1/rho DIP}(Y_h^k - U^k)
    def Z_update(self, Yh, U):
        Z = self.prox_operator(Yh - U)
        return Z

    def Yh_update(self, Z, U, D_transpose_Ys):        
        rhs = 2 * D_transpose_Ys + self.rho * (Z + U) # 2 * D^T * Ys + rho * (Z + U)

        # Model 2
        A = self.conv1x1_D(rhs)     # Downsample
        batch_size, channels, height, width = A.shape
        A_flat = A.permute(0, 2, 3, 1).reshape(-1, channels)
        Phi = self.channelFc(A_flat)
        N = Phi.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        B = self.conv1x1_U(N)       # Upsample
        Yh = (rhs - (2 / self.rho) * B)/self.rho
        return Yh
    
    # U^{k+1} = U^k - Y_h^k + Z^{k+1}
    def U_update(self, U, Yh, Z):
        return U - Yh + Z
    
    def forward(self, Ys, Yh, U):         
        batch_size, _, height, width = Ys.shape

        Ys_flat = Ys.reshape(batch_size, self.in_channels, -1) # [B, C_in, H, W] -> [B, C_in, H*W]        
        # Compute D^T * Y: [C_out, C_in] * [B, C_in, H*W] -> [B, C_out, H*W]
        D_transpose_Ys = torch.matmul(self.D.transpose(0, 1).squeeze(-1).squeeze(-1), Ys_flat)
        D_transpose_Ys = D_transpose_Ys.reshape(batch_size, self.out_channels, height, width) # [B, C_out, H*W] -> [B, C_out, H, W]
        
        for i in range(self.num_iterations):
            Z = self.Z_update(Yh, U)
            Yh = self.Yh_update(Z, U, D_transpose_Ys)
            U = self.U_update(U, Yh, Z)
            
        return Yh, U
        

class FinalStage(nn.Module):
    def __init__(self, shared_prox_operator):
        super(FinalStage, self).__init__()
        self.prox_operator = shared_prox_operator
    
    # Z^{k+1} = Prox_{1/rho DIP}(Y_h^k - U^k)
    def Z_update(self, Yh, U):
        Z = self.prox_operator(Yh - U)
        return Z

    def forward(self, Yh, U):           
        Z = self.Z_update(Yh, U)
            
        return Z

class COS2A(nn.Module):
    def __init__(self, in_channels=12, out_channels=172):
        super(COS2A, self).__init__()
        self.in_channels = in_channels      # MSI channels (c)
        self.out_channels = out_channels    # HSI channels (C)
        self.num_iterations = 2             #***#
        
        self.shared_D = nn.Parameter(torch.randn(in_channels, out_channels, 1, 1))
        nn.init.xavier_normal_(self.shared_D)        
        self.shared_prox_operator = ProximalOperator(out_channels)
        self.shared_channel_fc = SymmetricLinear(in_channels)
        
        self.init_stage = InitStage(self.shared_D, self.shared_prox_operator, self.shared_channel_fc, in_channels, out_channels)
        self.mid_stage = MidStage(self.shared_D, self.shared_prox_operator, self.shared_channel_fc, in_channels, out_channels, num_iterations=self.num_iterations)
        self.final_stage = FinalStage(self.shared_prox_operator)

    def forward(self, Ys):
        Yh1, U1 = self.init_stage(Ys)
        Yh2, U2 = self.mid_stage(Ys, Yh1, U1)
        Z_final = self.final_stage(Yh2, U2)
        return Z_final


if __name__ == '__main__':
    batch_size = 2
    in_channels = 12    # MSI channels
    out_channels = 172  # HSI channels
    height, width = 150, 150
    
    Ys = torch.randn(batch_size, in_channels, height, width, device=device)
    
    model = COS2A(in_channels, out_channels).to(device)
    
    Z_final = model(Ys)
    
    print(f"Input MSI shape: {Ys.shape}")
    print(f"Output HSI shape: {Z_final.shape}")
    print(f"D shape: {model.shared_D.shape}")
    print(f"D is trainable: {model.shared_D.requires_grad}")
    
    lower_tri_non_neg = F.softplus(model.shared_channel_fc.lower_triangular)
    init_W = torch.tril(lower_tri_non_neg) + torch.tril(lower_tri_non_neg, -1).T
    
    print(f"\nSymmetric matrix shape: {init_W.shape}")
    print(f"Is symmetric: {torch.allclose(init_W, init_W.T)}")
    
    # Check if the shared components are actually shared
    print("\nVerifying shared components:")
    print(f"Shared D between init and mid stages: {model.init_stage.D is model.mid_stage.D}")
    print(f"Shared ProximalOperator across stages: {model.init_stage.prox_operator is model.mid_stage.prox_operator is model.final_stage.prox_operator}")
    print(f"Shared SymmetricLinear between init and mid stages: {model.init_stage.channelFc is model.mid_stage.channelFc}")
    
    # Verify forward pass
    print("\nForward pass successful!")
