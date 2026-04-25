import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqInv(nn.Module):
    def __init__(self,r,vocab,dm=512,nhead=8,enc_layers=6,dec_layers=6,ff=2048,drop=0.1,max_T=128,n_mech=8,n_sig=8):
        super().__init__()
        self.in_proj=nn.Linear(r,dm)
        self.pos=nn.Embedding(max_T,dm)
        self.mech_emb=nn.Embedding(n_mech,dm)
        self.sig_emb=nn.Embedding(n_sig,dm)
        enc=nn.TransformerEncoderLayer(dm,nhead,ff,drop,batch_first=True)
        self.enc=nn.TransformerEncoder(enc,enc_layers)
        self.tok_emb=nn.Embedding(vocab,dm)
        self.tgt_pos=nn.Embedding(max_T,dm)
        dec=nn.TransformerDecoderLayer(dm,nhead,ff,drop,batch_first=True)
        self.dec=nn.TransformerDecoder(dec,dec_layers)
        self.head=nn.Linear(dm,vocab,bias=False)
        self.head.weight=self.tok_emb.weight
        self.dm=dm
        self.max_T=max_T
    def encode(self,z,mech_id,sig_id):
        B,T,r=z.shape
        x=self.in_proj(z)
        p=self.pos(torch.arange(T,device=z.device))[None]
        m=self.mech_emb(mech_id)[:,None,:]
        s=self.sig_emb(sig_id)[:,None,:]
        x=x+p+m+s
        return self.enc(x)
    def decode_step(self,mem,tgt_ids):
        B,L=tgt_ids.shape
        e=self.tok_emb(tgt_ids)
        p=self.tgt_pos(torch.arange(L,device=tgt_ids.device))[None]
        x=e+p
        causal=torch.triu(torch.ones(L,L,device=x.device,dtype=torch.bool),1)
        h=self.dec(x,mem,tgt_mask=causal)
        return self.head(h)
    def forward(self,z,mech_id,sig_id,tgt_ids):
        mem=self.encode(z,mech_id,sig_id)
        return self.decode_step(mem,tgt_ids)

def beam_search(model,z,mech_id,sig_id,bos,eos,max_len,B=8):
    model.eval()
    mem=model.encode(z,mech_id,sig_id)
    Bz=z.size(0)
    seqs=torch.full((Bz,1,1),bos,device=z.device,dtype=torch.long)
    scores=torch.zeros(Bz,1,device=z.device)
    for _ in range(max_len-1):
        Bz,K,L=seqs.shape
        flat=seqs.view(Bz*K,L)
        mem_e=mem.unsqueeze(1).expand(-1,K,-1,-1).reshape(Bz*K,mem.size(1),mem.size(2))
        logits=model.decode_step(mem_e,flat)[:,-1,:]
        lp=F.log_softmax(logits,dim=-1)
        cand=scores.unsqueeze(-1)+lp.view(Bz,K,-1)
        V=cand.size(-1)
        flat_c=cand.view(Bz,-1)
        top_s,top_i=flat_c.topk(B,dim=-1)
        prev=top_i//V
        tok=(top_i%V).unsqueeze(-1)
        old=torch.gather(seqs,1,prev.unsqueeze(-1).expand(-1,-1,L))
        seqs=torch.cat([old,tok],dim=-1)
        scores=top_s
    return seqs,scores

def mech_log_likelihood(quotient_model,frozen_lm,layer,prompt_ids,z_obs,sigma_rel):
    with torch.no_grad():
        h=frozen_lm(prompt_ids,output_hidden_states=True).hidden_states[layer]
        mu,ls=quotient_model.enc(h)
        var=ls.exp()+sigma_rel**2
        ll=-0.5*(((z_obs-mu)**2/var).sum(dim=-1)+var.log().sum(dim=-1))
    return ll.sum(dim=-1)
