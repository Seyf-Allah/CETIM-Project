"""Analyse a complete recording using the trained 1-second preprocessing."""
import argparse
from pathlib import Path
import torch
from stream_pipeline import load_audio,Streamer,compute_window_probs,decide
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('audio',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args();torch.set_num_threads(2)
    waveform,_=load_audio(a.audio)
    real_duration=len(waveform)/16000
    if len(waveform)<16000:waveform=torch.nn.functional.pad(waveform,(0,16000-len(waveform)))
    model=Streamer(device=a.device,n_threads=2)
    result=compute_window_probs(waveform,model,win_s=1.,hop_s=.5,pad_mode='train',batch_size=2)
    # No unvalidated post-filter, silence override, or severity policy.
    df=decide(result,silence_db=None)
    df['t_fin_s']=df['t_fin_s'].clip(upper=real_duration)
    destination=a.output or Path(__file__).resolve().parent/'predictions_latest.csv'
    destination.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(destination,index=False,encoding='utf-8-sig')
    print(df.to_string(index=False));print('CSV:',destination.resolve())
if __name__=='__main__':main()
