from pathlib import Path
import argparse,csv,sys
import numpy as np
import torch,torchaudio
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'compat'))
from backbone.configuration_eat import EATConfig
from backbone.modeling_eat import EATModel
class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=EATModel(EATConfig.from_json_file(str(ROOT/'backbone/config.json')))
        self.head=torch.nn.Sequential(torch.nn.LayerNorm(768),torch.nn.Dropout(.2),torch.nn.Linear(768,5))

    def forward(self,x):
        features=self.encoder.extract_features(x)
        if features.ndim==3:features=features[:,0,:]
        return self.head(features)


def load_model(device):
    saved=torch.load(ROOT/'modele.pt',map_location='cpu',weights_only=True)
    model=Classifier()
    model.load_state_dict(saved['state_dict'],strict=True)
    return model.to(device).eval(),saved['labels']


def fbank(wav):
    assert len(wav)==16000
    x=torch.as_tensor(wav,dtype=torch.float32).clone()
    x=x-x.mean()
    mel=torchaudio.compliance.kaldi.fbank(x[None],htk_compat=True,sample_frequency=16000,use_energy=False,
        window_type='hanning',num_mel_bins=128,dither=0.,frame_shift=10)
    pad=(-mel.shape[0])%16
    mel=F.pad(mel,(0,0,0,pad))
    return ((mel+4.268)/(4.569*2))[None]

def analyze(model,labels,wave,sr,device='cpu'):
    wave=np.asarray(wave,dtype='float32')
    if wave.ndim==2:wave=wave.mean(axis=1)
    if wave.ndim!=1 or not len(wave) or not np.isfinite(wave).all():raise ValueError('Audio vide ou non fini.')
    w=torch.from_numpy(wave.copy())
    if sr!=16000:w=torchaudio.functional.resample(w,int(sr),16000)
    w=w.numpy();rows=[]
    with torch.inference_mode():
        for end in range(16000,len(w)+1,8000):
            part=w[end-16000:end]
            if np.max(np.abs(part))==0:
                prob=np.eye(len(labels))[labels.index('negatif')]
            else:prob=model(fbank(part)[None].to(device)).softmax(-1)[0].cpu().numpy()
            rows.append(dict(context_start_s=(end-16000)/16000,context_end_s=end/16000,
                 target_start_s=(end-8000)/16000,target_end_s=end/16000,
                 label=labels[int(np.argmax(prob))],**{f'p_{name}':float(prob[i]) for i,name in enumerate(labels)}))
    return rows

def main():
 import soundfile as sf
 p=argparse.ArgumentParser(description='V7 officielle complete : entree 1 s, pas 0,5 s.')
 p.add_argument('audio',type=Path);p.add_argument('--output',type=Path,default=Path('predictions_v7.csv'));a=p.parse_args()
 torch.set_num_threads(4)
 x,sr=sf.read(a.audio,dtype='float32',always_2d=True)
 if len(x)<sr:p.error('Audio trop court : minimum une seconde reelle.')
 model,labels=load_model(torch.device('cpu'));rows=analyze(model,labels,x,sr,'cpu')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(f'{len(rows)} predictions : {a.output.resolve()}')
if __name__=='__main__':main()
