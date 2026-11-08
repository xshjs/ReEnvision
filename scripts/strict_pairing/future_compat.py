"""Scoped import bridge for legacy diffusers; never downgrade the original environment."""
from pathlib import Path

def prepare_tcn_checkpoint(source,destination):
    """Add ONLY disabled head keys required by the newer strict loader.

    Original delta weights and source checkpoint stay untouched. No trained
    validity/risk predictor is invented. All existing weights are byte-equal.
    """
    import torch
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    state=checkpoint['model_state']
    keys=['validity_proj.weight','validity_proj.bias','uncertainty_proj.weight','uncertainty_proj.bias']
    missing=[key for key in keys if key not in state]
    if not missing:return Path(source),dict(required=False)
    if len(missing)!=4 or checkpoint.get('has_validity_head',False):
        raise ValueError('Partially missing or declared trained validity heads; refusing to fabricate weights')
    hidden=int(checkpoint.get('hidden_dim',128));reference=state['in_proj.weight']
    if reference.shape[0]!=hidden:raise ValueError('TCN hidden dimension mismatch')
    converted=dict(checkpoint);converted['model_state']=dict(state)
    for key in keys:
        shape=(1,hidden,1) if key.endswith('weight') else (1,)
        converted['model_state'][key]=torch.zeros(shape,dtype=reference.dtype)
    converted['has_validity_head']=False
    converted['loader_compatibility']='Disabled auxiliary head keys only; all original delta parameters unchanged'
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    torch.save(converted,destination)
    return destination,dict(required=True,added_keys=keys,has_validity_head=False,original_parameters_unchanged=True)

def install():
    import huggingface_hub as hub
    if not hasattr(hub,'cached_download'):
        def cached_download(url,*args,**kwargs):
            # Legacy diffusers imports this symbol for dynamic remote modules.
            # Experiment uses pre-existing local models only. Never silently
            # download different weights to make an ablation succeed.
            path=Path(str(url))
            if path.is_file():return str(path)
            raise RuntimeError('Legacy cached_download remote loading disabled: provide the original local model folders')
        hub.cached_download=cached_download
