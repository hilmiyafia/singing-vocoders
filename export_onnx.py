import torch
from train import _train
import torch.onnx
from utils.training_utils import get_latest_checkpoint_path

class Wrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mel, f0):
        return self.model.Gforward({
            "mel": mel.permute(0, 2, 1), 
            "f0": f0})["audio"][:, 0]

def export(config, exp_name, work_dir):
    model = _train(config, exp_name, work_dir, False)

    data = torch.load(get_latest_checkpoint_path(work_dir + "/" + exp_name))
    model.setup("")
    model.load_state_dict(data["state_dict"])

    mel = torch.randn(1, 100, 128)
    f0 = torch.randn(1, 100)
    torch.onnx.export(
        Wrapper(model), 
        dynamo=False,
        args=(mel, f0), 
        input_names=["mel", "f0"],
        output_names=["waveform"],
        dynamic_axes={
            "mel": {1: "n_frames"},
            "f0": {1: "n_frames"},
            "waveform": {1: "n_samples"}},
        external_data=False,
        f="output/model.onnx")

if __name__ == "__main__":
    config = "output/vocoder/config.yaml"
    exp_name = "vocoder"
    work_dir = "output"
    export(config, exp_name, work_dir)
