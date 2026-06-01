import torch
import torch.nn.functional as F
import numpy as np
import quest.utils.tensor_utils as TensorUtils
import itertools

from quest.algos.base import ChunkPolicy


class QueST(ChunkPolicy):
    def __init__(self,
                 autoencoder,
                 policy_prior,
                 stage,
                 loss_fn,
                 l1_loss_scale,
                 action_target_key="actions",
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.autoencoder = autoencoder
        self.policy_prior = policy_prior
        self.stage = stage

        self.start_token = self.policy_prior.start_token
        self.l1_loss_scale = l1_loss_scale if stage == 2 else 0
        self.codebook_size = np.array(autoencoder.fsq_level).prod()
        self.action_target_key = action_target_key
        
        self.loss = loss_fn

    def _autoencoder_forward(self, data):
        if getattr(self.autoencoder, "use_ft_conditioning", False):
            return self.autoencoder(data["actions"], ft=data.get("masked_ft"))
        return self.autoencoder(data["actions"])

    def _autoencoder_get_indices(self, data):
        if getattr(self.autoencoder, "use_ft_conditioning", False):
            return self.autoencoder.get_indices(data["actions"], ft=data.get("masked_ft"))
        return self.autoencoder.get_indices(data["actions"])

    def _action_target(self, data):
        if self.action_target_key not in data:
            raise KeyError(f"Batch is missing action target key '{self.action_target_key}'.")
        return data[self.action_target_key]
        
    def get_optimizers(self):
        if self.stage == 0:
            decay, no_decay = TensorUtils.separate_no_decay(self.autoencoder)
            optimizers = [
                self.optimizer_factory(params=decay),
                self.optimizer_factory(params=no_decay, weight_decay=0.)
            ]
            return optimizers
        elif self.stage == 1:
            decay, no_decay = TensorUtils.separate_no_decay(self, 
                                                            name_blacklist=('autoencoder',))
            optimizers = [
                self.optimizer_factory(params=decay),
                self.optimizer_factory(params=no_decay, weight_decay=0.)
            ]
            return optimizers
        elif self.stage == 2:
            decay, no_decay = TensorUtils.separate_no_decay(self, 
                                                            name_blacklist=('autoencoder',))
            decoder_decay, decoder_no_decay = TensorUtils.separate_no_decay(self.autoencoder.decoder)
            optimizers = [
                self.optimizer_factory(params=itertools.chain(decay, decoder_decay)),
                self.optimizer_factory(params=itertools.chain(no_decay, decoder_no_decay), weight_decay=0.)
            ]
            return optimizers

    def get_context(self, data):
        obs_emb = self.obs_encode_tokens(data)
        task_emb = self.get_task_emb(data).unsqueeze(1)
        context = torch.cat([task_emb, obs_emb], dim=1)
        return context

    def compute_loss(self, data):
        if self.stage == 0:
            return self.compute_autoencoder_loss(data)
        elif self.stage == 1:
            return self.compute_prior_loss(data)
        elif self.stage == 2:
            return self.compute_prior_loss(data)

    def compute_autoencoder_loss(self, data):
        pred, pp, pp_sample, aux_loss, _ = self._autoencoder_forward(data)
        target = self._action_target(data)
        recon_loss = self.loss(pred, target)
        if self.autoencoder.vq_type == 'vq':
            loss = recon_loss + aux_loss
        else:
            loss = recon_loss

        with torch.no_grad():
            gripper_idx = data["actions"].shape[-1] - 1
            target_gripper = data["actions"][..., gripper_idx]
            pred_gripper = pred[..., gripper_idx]
            target_gripper_active = target_gripper.abs() > 0.5
            pred_gripper_active = pred_gripper.abs() > 0.5
            gripper_active_count = target_gripper_active.sum()
            if gripper_active_count > 0:
                gripper_recall = (
                    (pred_gripper_active & target_gripper_active).sum().float()
                    / gripper_active_count.float()
                )
            else:
                gripper_recall = torch.tensor(0.0, device=pred.device)
            
        info = {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'aux_loss': aux_loss.sum().item(),
            'pp': pp.item(),
            'pp_sample': pp_sample.item(),
            'gripper_recall_when_abs_gt_0_5': gripper_recall.item(),
            'gripper_abs_gt_0_5_count': gripper_active_count.item(),
        }
        return loss, info

    def compute_prior_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        with torch.no_grad():
            indices = self._autoencoder_get_indices(data).long()
        context = self.get_context(data)
        start_tokens = (torch.ones((context.shape[0], 1), device=self.device, dtype=torch.long) * self.start_token)
        x = torch.cat([start_tokens, indices[:,:-1]], dim=1)
        targets = indices.clone()
        logits = self.policy_prior(x, context)
        prior_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        with torch.no_grad():
            logits = logits[:,:,:self.codebook_size]
            probs = torch.softmax(logits, dim=-1)
            sampled_indices = torch.multinomial(probs.view(-1,logits.shape[-1]),1)
            sampled_indices = sampled_indices.view(-1,logits.shape[1])
        
        pred_actions = self.autoencoder.decode_actions(sampled_indices)
        l1_loss = self.loss(pred_actions, self._action_target(data))
        total_loss = prior_loss + self.l1_loss_scale * l1_loss
        info = {
            'loss': total_loss.item(),
            'nll_loss': prior_loss.item(),
            'l1_loss': l1_loss.item()
        }
        return total_loss, info

    def sample_actions(self, data):
        data = self.preprocess_input(data, train_mode=False)
        context = self.get_context(data)
        sampled_indices = self.policy_prior.get_indices_top_k(context, self.codebook_size)
        pred_actions = self.autoencoder.decode_actions(sampled_indices)
        pred_actions = pred_actions.permute(1,0,2)
        return pred_actions.detach().cpu().numpy()
