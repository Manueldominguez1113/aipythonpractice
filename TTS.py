import torch
from scipy.io.wavfile import write


cuda_is_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_is_available else "cpu")
# device = torch.load('models/',map_location ='cpu')
text = "hello world, I missed you so much. I love you."
torch.hub.set_dir("./models")


def load_waveglow():
    waveglow = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub","nvidia_waveglow")
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to(device)
    waveglow.eval()
    return waveglow

waveglow = load_waveglow()
# def load_tacotron():
taco = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub","nvidia_tacotron2",model_math="fp32")
taco = taco.to(device)
taco.eval()

utils = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub","nvidia_tts_utils")
sequences, lengths = utils.prepare_input_sequence([text])

with torch.no_grad():
    mel, _, _ = taco.infer(sequences, lengths)
    audio = waveglow.infer(mel)
    audio_numpy = audio[0].data.cpu().numpy()
    rate = 22050
    write("audio.wav", rate, audio_numpy)

# def load_waveglow():
#     waveglow = torch.load("NVIDIA/DeepLearningExamples:torchhub","waveglow", trust_repo= True)
#     waveglow = waveglow.remove_weightnorm(waveglow)
#     waveglow = waveglow.to(device)
#     waveglow.eval()
#     return waveglow
#
#
# taco = torch.load("NVIDIA/DeepLearningExamples:torchhub","tacotron2",model_math="fp16",trust_repo=True)
# taco = taco.to(device)
# taco.eval()