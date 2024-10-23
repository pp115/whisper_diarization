import whisper
from pyannote.audio import Pipeline
from whisper_utils import diarize_text
from pydub import AudioSegment

# sound = AudioSegment.from_mp3("../data/Frog_Star_Trek_NPR.mp3")
# sound.export("../data/Frog_Star_Trek_NPR.wav", format="wav")

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token="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")
model = whisper.load_model("tiny.en")
asr_result = model.transcribe("../data/Frog_Star_Trek_NPR.wav")
diarization_result = pipeline("../data/Frog_Star_Trek_NPR.wav")
final_result = diarize_text(asr_result, diarization_result)

for seg, spk, sent in final_result:
    line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}'
    print(line)

import importlib
print(importlib.__version__)

config = utils.backends._set_configs_from_environment()
