import torch.utils.data.distributed
import torch.utils.data.distributed
from model import Timesformer,C3D
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    device = torch.device('cpu')
    model_pt = 'C3D.pt'

    model = C3D()
    model.load_state_dict(torch.load('checkpoints/C3D/checkpoint_10.pth', map_location=device))


    model.to(device)
    model.eval()

    input_tensor = torch.rand(1, 30, 3, 112, 112)

    traced_script_module = torch.jit.trace(model, input_tensor.to(torch.float32))
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter("./scriptmodel/C3D.ptl")

