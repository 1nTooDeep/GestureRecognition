import torch.utils.data.distributed
from model.model import C3D_L

if __name__ == "__main__":
    device = torch.device('cpu')
    model_ori_pt = 'C3D_ori.pt'
    model_pruned_pt = 'C3D_pruned.pt'

    model_ori = C3D_L()
    model_pruned = C3D_L()
    model_ori.load_state_dict(torch.load('mbin/C3D.pth', map_location=device))
    model_pruned.load_state_dict(torch.load('mbin/C3D.pth', map_location=device))


    model_ori.to(device)
    model_pruned.to(device)
    model_ori.eval()
    model_pruned.eval()

    input_tensor = torch.rand(1, 24, 30, 112, 112)

    mobile_ori = torch.jit.trace(model_ori, input_tensor)
    model_pruned = torch.jit.trace(model_pruned, input_tensor)
    mobile_ori.save(model_ori_pt)
    model_pruned.save(model_pruned_pt)