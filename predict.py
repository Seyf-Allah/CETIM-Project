"""Standalone SSLAM V6 inference: python predict.py recording.wav"""
import argparse
import json
from pathlib import Path
import torch
from backbone.configuration_eat import EATConfig
from backbone.modeling_eat import EATModel
from preprocessing import audio_features

ROOT=Path(__file__).resolve().parent


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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio',type=Path)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args()
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    torch.set_num_threads(2)
    model,labels=load_model(device)
    with torch.inference_mode():
        scores=model(audio_features(args.audio).unsqueeze(0).to(device)).softmax(-1)[0].cpu().tolist()
    print(json.dumps(dict(classe=labels[max(range(5),key=lambda i:scores[i])],scores=dict(zip(labels,scores)),device=str(device)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
