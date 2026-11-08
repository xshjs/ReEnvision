"""Matched frozen-backbone future/current controls, isolated from production inference."""
import ast
import copy
import functools
import inspect
import textwrap
import torch


def observed_latents(extractor,image_latents,noise,timestep):
    """Denoise an observed current image, NOT an unknown future latent.

    prepare_latents has already drawn the same full 16-frame noise tensor as
    predicted mode, preserving the RNG stream. Only its first slot is used.
    SVD image-condition latents are unscaled; diffusion latents are VAE-scaled.
    """
    clean=image_latents*extractor.pipeline.vae.config.scaling_factor
    standard_noise=noise[:,:1]/extractor.pipeline.scheduler.init_noise_sigma
    ts=timestep.reshape(1).expand(clean.shape[0])
    return extractor.pipeline.scheduler.add_noise(clean,standard_noise,ts)


def single_frame_forward(original):
    """Derive a one-observed-frame version from the installed extractor source."""
    unwrapped=inspect.unwrap(original)
    tree=ast.parse(textwrap.dedent(inspect.getsource(unwrapped)))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef));fn.decorator_list=[]
    changed={'frames':0,'noise':0}
    class Modify(ast.NodeTransformer):
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id=='num_frames' for t in node.targets):
                node.value=ast.Constant(1);changed['frames']+=1
            if any(isinstance(t,ast.Name) and t.id=='latents' for t in node.targets) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='prepare_latents':
                # Preserve the ORIGINAL full-sized noise draw, even though current
                # features have no unobserved temporal slots.
                node.value.args[1]=ast.Constant(16);changed['noise']+=1
                return [node,*ast.parse('latents = _observed_latents(self, image_latents, latents, timesteps[0])').body]
            return self.generic_visit(node)
    Modify().visit(fn)
    if changed!={'frames':1,'noise':1}:raise ValueError(f'Unsupported feature extractor layout: {changed}')
    ns=dict(unwrapped.__globals__);ns['_observed_latents']=observed_latents
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    exec(compile(module,inspect.getsourcefile(unwrapped),'exec'),ns)
    return torch.no_grad()(ns[fn.name])


def install_feature_control(extractor_class,mode):
    if mode not in {'predicted','current_observed','current_slot'}:raise ValueError(mode)
    if mode=='predicted':return  # Bit-for-bit original method; no wrapper at all.
    if getattr(extractor_class.forward,'_future_ablation',False):raise RuntimeError('One feature condition per process')
    original=extractor_class.forward
    current=single_frame_forward(original) if mode=='current_observed' else None
    @functools.wraps(original)
    def forward(self,*args,**kwargs):
        features=(current if current is not None else original)(self,*args,**kwargs)
        if features.ndim!=5 or not torch.isfinite(features).all():raise ValueError('Invalid observed feature tensor')
        # current_slot is only a weaker diagnostic: it still runs the full SVD.
        return features[:,:1].expand(-1,16,-1,-1,-1).clone()
    forward._future_ablation=True
    extractor_class.forward=forward
