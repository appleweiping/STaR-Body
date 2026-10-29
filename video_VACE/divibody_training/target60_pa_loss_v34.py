"""FP32 train-only detached-transform Procrustes auxiliary loss."""
import torch
def detached_similarity_align(prediction,target,eps=1e-8):
    if prediction.shape!=target.shape or prediction.ndim!=3 or prediction.shape[-1]!=3:raise ValueError('Expected matched [N,J,3]')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():raise FloatingPointError('Nonfinite PA input')
    with torch.no_grad():
        source=prediction.detach().float().transpose(1,2);dest=target.detach().float().transpose(1,2)
        sm=source.mean(2,keepdim=True);dm=dest.mean(2,keepdim=True)
        xs=source-sm;ys=dest-dm
        variance=xs.square().sum((1,2));target_variance=ys.square().sum((1,2))
        if (variance<=eps).any() or (target_variance<=eps).any():raise ValueError('Degenerate PA training pose; do not silently remove owned frames')
        cov=xs@ys.transpose(-1,-2);u,_,vh=torch.linalg.svd(cov)
        correction=torch.eye(3,device=source.device).expand(len(source),-1,-1).clone()
        correction[:,-1,-1]=torch.where(torch.linalg.det(u@vh)<0,-1.,1.)
        rotation=vh.transpose(-1,-2)@correction@u.transpose(-1,-2)
        scale=torch.diagonal(rotation@cov,dim1=-2,dim2=-1).sum(-1)/variance
        translation=dm-scale[:,None,None]*(rotation@sm)
        if not torch.isfinite(scale).all() or not torch.isfinite(rotation).all() or not (torch.linalg.det(rotation)>.999).all():raise FloatingPointError('Invalid fitted PA transform')
    return (scale[:,None,None]*(rotation@prediction.float().transpose(1,2))+translation).transpose(1,2)
def pa_auxiliary_loss(prediction,batch,models,regressor,device):
    from divibody_training.train_divibody_continuous_geometry_v5 import camera_targets,smpl_vertices,gendered_vertices
    from divibody_training.geometry_metrics_3dpw import pelvis_aligned_j14
    mask=(batch['frame_owner']&batch['target_valid']).to(device).reshape(-1)
    if not mask.any():raise ValueError('PA auxiliary requires owned training frames')
    with torch.autocast(device_type=device.type,enabled=False):
        r=prediction['rotmat'].float().reshape(-1,24,3,3)[mask]
        pv=smpl_vertices(models[2],r,prediction['betas'].float().reshape(-1,10)[mask])
        with torch.no_grad():
            gv=gendered_vertices(models,camera_targets(batch,device).reshape(-1,24,3,3)[mask],
                batch['betas'].to(device,dtype=torch.float32).reshape(-1,10)[mask],
                batch['gender_code'].to(device).reshape(-1)[mask])
        pj,_=pelvis_aligned_j14(pv,regressor);gj,_=pelvis_aligned_j14(gv,regressor)
        aligned=detached_similarity_align(pj,gj)
        return torch.linalg.vector_norm(aligned-gj,dim=-1).mean()
