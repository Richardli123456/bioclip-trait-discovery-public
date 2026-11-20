"""Train a small TopK SAE; select by validation genera and evaluate test genera once."""
from pathlib import Path
from collections import Counter
import csv,copy,json,time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent

class TopKSAE(nn.Module):
    def __init__(self,mean,scale,width=2048,k=32):
        super().__init__();self.k=k;self.width=width
        self.register_buffer('mean',mean.clone());self.register_buffer('scale',scale.clone())
        self.encoder=nn.Linear(768,width)
        self.decoder=nn.Linear(width,768,bias=False)
        self.output_bias=nn.Parameter(torch.zeros(768,device=mean.device))
        with torch.no_grad():
            self.decoder.weight.copy_(F.normalize(self.decoder.weight,dim=0))
            self.encoder.weight.copy_(self.decoder.weight.T)
            self.encoder.bias.zero_()
    def encode(self,raw):
        positive=F.relu(self.encoder((raw-self.mean)/self.scale))
        values,indices=positive.topk(self.k,dim=-1)
        return torch.zeros_like(positive).scatter(-1,indices,values)
    def decode(self,z):return (self.decoder(z)+self.output_bias)*self.scale+self.mean
    def forward(self,raw):
        z=self.encode(raw);return self.decode(z),z
    @torch.no_grad()
    def normalize_decoder(self):self.decoder.weight.copy_(F.normalize(self.decoder.weight,dim=0))

def metrics(x,y,z,mean):
    delta=(x-y).double();centered=(x-mean).double()
    nmse=float(delta.square().sum()/centered.square().sum())
    return dict(mse=float(delta.square().mean()),normalized_mse=nmse,
        variance_explained_vs_training_mean=1-nmse,
        cosine_mean=float(F.cosine_similarity(x,y,dim=-1).mean()),
        cosine_p05=float(torch.quantile(F.cosine_similarity(x,y,dim=-1),.05)),
        mean_active_features=float((z>1e-7).float().sum(-1).mean()),
        active_dictionary_features=int((z>1e-7).any(0).sum()),
        dead_dictionary_features=int((~(z>1e-7).any(0)).sum()))

def main():
    start=time.perf_counter();torch.manual_seed(20260929);np.random.seed(20260929)
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    rows=json.loads((ROOT/'data/species.json').read_text())
    raw=torch.from_numpy(np.load(ROOT/'data/embeddings_raw.npy')).cuda()
    idx={s:torch.tensor([i for i,r in enumerate(rows) if r['split']==s],device='cuda') for s in ['train','validation','test']}
    data={s:raw[v] for s,v in idx.items()}
    mean=data['train'].mean(0);scale=(data['train']-mean).square().mean().sqrt()
    model=TopKSAE(mean,scale).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    best=float('inf');best_state=None;best_epoch=0;history=[];no_improve=0
    for epoch in range(1,201):
        model.train();order=torch.randperm(len(data['train']),device='cuda');total=0
        for batch in order.split(256):
            x=data['train'][batch];pred,z=model(x)
            loss=((pred-x)/scale).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step();model.normalize_decoder();total+=float(loss.detach())*len(x)
        if epoch==1 or epoch%5==0:
            model.eval()
            with torch.inference_mode():
                pred,z=model(data['validation']);val=metrics(data['validation'],pred,z,mean)
            history.append(dict(epoch=epoch,train_normalized_mse=total/len(data['train']),
                                validation_normalized_mse=val['normalized_mse']))
            if val['normalized_mse']<best-1e-5:
                best=val['normalized_mse'];best_epoch=epoch
                best_state={k:v.detach().clone() for k,v in model.state_dict().items()};no_improve=0
            else:no_improve+=1
            if epoch%20==0 or epoch==1:print('Epoch',epoch,'validation NMSE',round(val['normalized_mse'],5),flush=True)
            if no_improve>=8:break
    model.load_state_dict(best_state);model.eval()
    with torch.inference_mode():
        reconstruction,z=model(raw)
        assert z.shape==(5388,2048) and bool(torch.isfinite(z).all())
        assert int((z>0).sum(-1).max())<=32
        scores={s:metrics(raw[i],reconstruction[i],z[i],mean) for s,i in idx.items()}
        # Training-only PCA basis; test data never influences fitting or checkpoint selection.
        _,_,vh=torch.linalg.svd(data['train']-mean,full_matrices=False)
        baselines={}
        for n in [32,128]:
            v=vh[:n].T;y=(raw-mean)@v@v.T+mean
            baselines[f'pca_{n}']={s:metrics(raw[i],y[i],torch.ones(len(i),n,device='cuda'),mean)
                                   for s,i in idx.items()}
    np.save(ROOT/'results/sparse_vectors.npy',z.cpu().numpy())
    np.save(ROOT/'results/reconstructed_embeddings.npy',reconstruction.cpu().numpy())
    torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},
                'config':{'input_dim':768,'width':2048,'k':32,'seed':20260929},
                'best_epoch':best_epoch},ROOT/'results/sae.pt')
    zn=z.cpu().numpy();active=zn>1e-7
    feature_rows=[]
    train_ids=idx['train'].cpu().numpy()
    for j in range(2048):
        ranked=train_ids[np.argsort(-zn[train_ids,j],kind='stable')[:8]]
        top=[{**rows[i],'activation':float(zn[i,j]),'row_index':int(i)} for i in ranked if zn[i,j]>1e-7]
        feature_rows.append(dict(feature=j,train_support=int(active[train_ids,j].sum()),
            total_support=int(active[:,j].sum()),top_training_species=top,
            top_genera=len({r['genus'] for r in top}),top_families=len({r['family'] for r in top})))
    (ROOT/'results/features.json').write_text(json.dumps(feature_rows,indent=2),encoding='utf-8')
    with (ROOT/'results/training_history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    evaluation=dict(architecture='768 -> 2048 nonnegative TopK (k=32) -> 768; linear decoder',
        preprocessing='Subtract training mean and divide by one training-only global RMS; inverse in decoder',
        selection='Lowest validation NMSE, checked every five epochs; test genera not used in selection',
        best_epoch=best_epoch,epochs_run=epoch,seed=20260929,batch_size=256,learning_rate=.001,
        metrics=scores,baselines=baselines,elapsed_seconds=time.perf_counter()-start,
        definition='NMSE = sum squared reconstruction error / sum squared distance to TRAINING mean; explained fraction=1-NMSE',
        caveats=['TopK fixes an upper bound of 32 active features; sparsity is imposed, not independently discovered.',
                 'PCA comparison is a linear reconstruction reference, not a parameter-matched interpretability benchmark.',
                 'Single SAE initialization; feature stability across seeds untested.',
                 'Taxonomic strings explicitly contain relatedness; feature families can reflect words or taxonomy instead of separable visual traits.'])
    (ROOT/'results/evaluation.json').write_text(json.dumps(evaluation,indent=2),encoding='utf-8')
    print(json.dumps(evaluation,indent=2),flush=True)

if __name__=='__main__':main()
