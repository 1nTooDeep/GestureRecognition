import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from model import Timesformer

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    device = torch.device('cpu')

    num_frames = 30
    model = Timesformer(
        image_size=160,
        num_frames=num_frames,
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=128
    )  # 5.1 0.30
    model.load_state_dict(torch.load('checkpoints/timesformer-14/checkpoint_21.pth', map_location=device))


    model.to(device)
    model.eval()

    input_tensor = torch.rand(1, 30, 3, 160, 160)

    traced_script_module = torch.jit.trace(model, input_tensor.to(torch.float32))
    # traced_script_module = torch.jit.script(model, input_tensor.to(torch.float32))
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter("./scriptmodel/timesformer-14.pt")

