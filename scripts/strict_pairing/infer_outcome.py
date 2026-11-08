"""Load outcome TCN and return explicit executor decisions, never execute a robot."""
import numpy as np
import torch
from outcome import OutcomeTCN,decide

class SelectiveRepair:
    def __init__(self,checkpoint,device='cpu'):
        self.device=device
        # Only load locally trusted checkpoints: torch pickle can execute code.
        self.ckpt=torch.load(checkpoint,map_location=device,weights_only=False)
        self.model=OutcomeTCN(self.ckpt['config']).to(device).eval()
        self.model.load_state_dict(self.ckpt['state_dict'])

    @torch.inference_mode()
    def predict(self,state,belief,action,delta_belief,task,meta,valid_steps,**gate):
        a=np.asarray(action,dtype=np.float32)
        if a.ndim!=2 or a.shape[1]!=7 or not 1<=len(a)<=10 or not 1<=valid_steps<=len(a):
            raise ValueError('Require aligned nonempty suffix of <=10 actions and valid mask')
        padded=np.pad(a,((0,10-len(a)),(0,0)))
        values=dict(state=state,belief=belief,delta_belief=delta_belief,action=padded,task=task,meta=meta,
            mask=(np.arange(10)<valid_steps).astype(np.float32))
        for k,n in self.ckpt['normalization'].items():values[k]=(np.asarray(values[k])-n['mean'])/n['std']
        tensors=[torch.as_tensor(values[k],dtype=torch.float32,device=self.device).unsqueeze(0)
            for k in ['state','belief','action','delta_belief','task','meta','mask']]
        h=self.model(*tensors);p=h['outcome_logits'].softmax(-1)[0].cpu().numpy()
        risk=float(h['risk_logit'].sigmoid()[0]);delta=h['delta'][0,:len(a)].cpu().numpy()
        delta[valid_steps:]=0
        # Caller cannot bypass an unvalidated checkpoint by passing a gate flag.
        gate.pop('thresholds_validated',None)
        validated=bool(self.ckpt.get('thresholds_validated',False))
        cap=self.ckpt.get('max_action_delta')
        if cap is None:validated=False
        elif gate.get('max_delta',cap)>cap:raise ValueError('Gate limit exceeds measured training candidate limit')
        gate.setdefault('max_delta',cap or .015)
        result=decide(a,delta,p,risk,thresholds_validated=validated,**gate)
        result.update(probabilities=p.tolist(),harmful_risk=risk)
        return result
