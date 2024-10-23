import os
import whisper
from pyannote.audio import Pipeline
from whisper_utils import diarize_text
from pydub import AudioSegment
import torch
import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

hf_token = os.environ['HF_TOKEN']

# sound = AudioSegment.from_mp3("../data/Frog_Star_Trek_NPR.mp3")
# sound.export("../data/Frog_Star_Trek_NPR.wav", format="wav")

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token="")
model = whisper.load_model("tiny.en")
asr_result = model.transcribe("../data/Frog_Star_Trek_NPR.wav")
diarization_result = pipeline("../data/Frog_Star_Trek_NPR.wav")
final_result = diarize_text(asr_result, diarization_result)

with open('../data/output.txt', 'w') as file:
    for seg, spk, sent in final_result:
        start = str(datetime.timedelta(seconds=seg.start)).split(".")[0]
        end = str(datetime.timedelta(seconds=seg.end)).split(".")[0]
        line = f'{start} {end} {spk} {sent}'
        file.write(line + "\n")
        print(line)

