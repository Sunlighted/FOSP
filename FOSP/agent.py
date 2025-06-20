import embodied
import jax
import jax.numpy as jnp
import ruamel.yaml as yaml
from distrax import Categorical
import copy
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
  def filter(self, record):
    return 'check_types' not in record.getMessage()
logger.addFilter(CheckTypesFilter())

from . import behaviors
from . import jaxagent
from . import jaxutils
from . import nets
from . import ninjax as nj

def symexp(x):
  return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)

@jaxagent.Wrapper
class Agent(nj.Module):

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    self.config = config
    self.obs_space = obs_space
    if isinstance(act_space, dict):
      self.act_space = act_space['action']
    else:
      self.act_space = act_space
    self.step = step
    self.wm = WorldModel(obs_space, act_space, config, name='wm')
    self.task_behavior = getattr(behaviors, config.task_behavior)(
        self.wm, self.act_space, self.obs_space, self.step, act_space, self.config, name='task_behavior')
    if config.expl_behavior == 'None':
      self.expl_behavior = self.task_behavior
    else:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.wm, self.act_space, self.config, name='expl_behavior')

  def policy_initial(self, batch_size):
    return (
        self.wm.initial(batch_size),
        self.task_behavior.initial(batch_size),
        self.expl_behavior.initial(batch_size))

  def train_initial(self, batch_size):
    return self.wm.initial(batch_size)

  def policy(self, obs, state, mode='train'):
    self.config.jax.jit and print('Tracing policy function.')
    obs = self.preprocess(obs)
    (prev_latent, prev_action), task_state, expl_state = state
    embed = self.wm.encoder(obs)
    latent, _ = self.wm.rssm.obs_step(
        prev_latent, prev_action, embed, obs['is_first'])
    task_outs, task_state = self.task_behavior.policy(latent, task_state)
    expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)
    if mode == 'eval':
      outs = task_outs
      outs['action'] = outs['action'].sample(seed=nj.rng())
    elif mode == 'explore':
      outs = expl_outs
      outs['log_entropy'] = outs['action'].entropy()
      outs['action'] = outs['action'].sample(seed=nj.rng())
    elif mode == 'train':
      outs = task_outs
      outs['log_entropy'] = outs['action'].entropy()

    state = ((latent, outs['action']), task_state, expl_state)
    return outs, state

  def train(self, data, state):
    self.config.jax.jit and print('Tracing train function.')
    metrics = {}
    data = self.preprocess(data)
    state, wm_outs, mets = self.wm.train(data, state)
    metrics.update(mets)
    context = {**data, **wm_outs['post']}
    start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
    _, mets = self.task_behavior.train(self.wm.imagine, start, context)
    metrics.update(mets)
    if self.config.expl_behavior != 'None':
      _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    outs = {}
    return outs, state, metrics

  def report(self, data):
    self.config.jax.jit and print('Tracing report function.')
    data = self.preprocess(data)
    report = {}
    report.update(self.wm.report(data))
    mets = self.task_behavior.report(data)
    report.update({f'task_{k}': v for k, v in mets.items()})
    if self.expl_behavior is not self.task_behavior:
      mets = self.expl_behavior.report(data)
      report.update({f'expl_{k}': v for k, v in mets.items()})
    return report

  def report_eval(self, data):
    self.config.jax.jit and print('Tracing report function.')
    data = self.preprocess(data)
    report = {}
    report.update(self.wm.report_eval(data))
    return report

  def preprocess(self, obs):
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      else:
        value = value.astype(jnp.float32)
      obs[key] = value
    obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
    return obs

def cost_from_state(wm, state):
  recon = wm.heads['decoder'](state)
  recon = recon['observation'].mode()
  hazards_size = 0.25
  batch_size = recon.shape[0] * recon.shape[1]
  hazard_obs = recon[:, :, 9:25].reshape(batch_size, -1, 2)
  hazards_dist = jnp.sqrt(jnp.sum(jnp.square(hazard_obs), axis=2)).reshape(
      batch_size,
      -1,
  )
  condition = jnp.less_equal(hazards_dist, hazards_size)
  cost = jnp.where(condition, 1.0, 0.0)
  cost = cost.sum(1)
  cost = cost.reshape(recon.shape[0], recon.shape[1])
  return cost

class WorldModel(nj.Module):

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    if isinstance(act_space, dict):
      self.act_space = act_space['action']
    else:
      self.act_space = act_space
    self.config = config
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
    self.encoder = nets.MultiEncoder(shapes, **config.encoder, name='enc')
    self.rssm = nets.RSSM(**config.rssm, name='rssm')
    self.heads = {
        'decoder': nets.MultiDecoder(shapes, **config.decoder, name='dec'),
        'reward': nets.MLP((), **config.reward_head, name='rew'),
        'cont': nets.MLP((), **config.cont_head, name='cont')}
    if self.config.use_cost:
      self.heads['cost'] = nets.MLP((), **config.cost_head, name='cost')
    self.opt = jaxutils.Optimizer(name='model_opt', **config.model_opt)
    scales = self.config.loss_scales.copy()
    image, vector = scales.pop('image'), scales.pop('vector')
    scales.update({k: image for k in self.heads['decoder'].cnn_shapes})
    scales.update({k: vector for k in self.heads['decoder'].mlp_shapes})
    self.scales = scales

  def initial(self, batch_size):
    prev_latent = self.rssm.initial(batch_size)
    prev_action = jnp.zeros((batch_size, *self.act_space.shape))
    return prev_latent, prev_action

  def train(self, data, state):
    modules = [self.encoder, self.rssm, *self.heads.values()]
    mets, (state, outs, metrics) = self.opt(
        modules, self.loss, data, state, has_aux=True)
    metrics.update(mets)
    return state, outs, metrics

  def loss(self, data, state):
    embed = self.encoder(data)
    prev_latent, prev_action = state
    prev_actions = jnp.concatenate([
        prev_action[:, None], data['action'][:, :-1]], 1)
    post, prior = self.rssm.observe(
        embed, prev_actions, data['is_first'], prev_latent)
    dists = {}
    feats = {**post, 'embed': embed}
    for name, head in self.heads.items():
      out = head(feats if name in self.config.grad_heads else sg(feats))
      out = out if isinstance(out, dict) else {name: out}
      dists.update(out)
    losses = {}
    losses['dyn'] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
    losses['rep'] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
    for key, dist in dists.items():
      if key == 'cost':
        condition = jnp.greater_equal(data['cost'], 1.0)
        loss = -dist.log_prob(data['cost'].astype(jnp.float32))
        loss = jnp.where(condition, self.config.cost_weight * loss, loss)
      else:
        loss = -dist.log_prob(data[key].astype(jnp.float32))
      assert loss.shape == embed.shape[:2], (key, loss.shape)
      losses[key] = loss
    scaled = {k: v * self.scales[k] for k, v in losses.items()}
    model_loss = sum(scaled.values())
    out = {'embed':  embed, 'post': post, 'prior': prior}
    out.update({f'{k}_loss': v for k, v in losses.items()})
    last_latent = {k: v[:, -1] for k, v in post.items()}
    last_action = data['action'][:, -1]
    state = last_latent, last_action
    metrics = self._metrics(data, dists, post, prior, losses, model_loss)
    return model_loss.mean(), (state, out, metrics)

  def imagine(self, policy, start, horizon, use_planner=False):
    first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
    keys = list(self.rssm.initial(1).keys())
    if use_planner:
      keys += ['action_mean','action_std', 'lagrange_multiplier', 'penalty_multiplier']
      start = {k: v for k, v in start.items() if k in keys}
      start['action'] = policy(start,0)
      def step(prev, current_horizon): # add the current_horizon
        prev = prev.copy()
        action_mean = prev['action_mean']
        action_std = prev['action_std']
        state = self.rssm.img_step(prev, prev.pop('action'))
        return {**state, 'action_mean':action_mean, 'action_std':action_std, 'action': policy(prev,current_horizon+1)}
    else:
      start = {k: v for k, v in start.items() if k in keys}
      start['action'] = policy(start)
      def step(prev, _):
        prev = prev.copy()
        state = self.rssm.img_step(prev, prev.pop('action'))
        return {**state, 'action': policy(state)}
    traj = jaxutils.scan(
        step, jnp.arange(horizon), start, self.config.imag_unroll)
    traj = {
        k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
    cont = self.heads['cont'](traj).mode()
    traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
    discount = 1 - 1 / self.config.horizon
    traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
    return traj

  def report(self, data):
    state = self.initial(len(data['is_first']))
    report = {}
    report.update(self.loss(data, state)[-1][-1])
    context, _ = self.rssm.observe(
        self.encoder(data)[:6, :5], data['action'][:6, :5],
        data['is_first'][:6, :5])
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    openl = self.heads['decoder'](
        self.rssm.imagine(data['action'][:6, 5:], start))
    for key in self.heads['decoder'].cnn_shapes.keys():
      truth = data[key][:6].astype(jnp.float32)
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      error = (model - truth + 1) / 2
      video = jnp.concatenate([truth, model, error], 2)
      report[f'openl_{key}'] = jaxutils.video_grid(video)
    return report

  def report_eval(self, data_expand):
    state = self.initial(len(data_expand['is_first']))
    report = {}
    report.update(self.loss(data_expand, state)[-1][-1])
    context, _ = self.rssm.observe(
        self.encoder(data_expand)[:, :5], data_expand['action'][:, :5],
        data_expand['is_first'][:, :5])
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    openl = self.heads['decoder'](
        self.rssm.imagine(data_expand['action'][:, 5:], start))
    for key in self.heads['decoder'].cnn_shapes.keys():
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      report[f'openl_{key}'] = jaxutils.video_grid(model)
    for key in self.heads['decoder'].mlp_shapes.keys():
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      report[f'openl_{key}'] = model
      if 'openl_observation' in report.keys() and not self.config.use_cost:
        report[f'openl_cost'] = self.cost_from_recon(report['openl_observation'])
    return report

  def cost_from_recon(self, recon):
    # jax format
    hazards_size = 0.25
    batch_size = recon.shape[0] * recon.shape[1]
    hazard_obs = recon[:, :, 9:25].reshape(batch_size, -1, 2)
    hazards_dist = jnp.sqrt(jnp.sum(jnp.square(hazard_obs), axis=2)).reshape(
        batch_size,
        -1,
    )

    condition = jnp.less_equal(hazards_dist, hazards_size)
    cost = jnp.where(condition, 1.0, 0.0)
    cost = cost.sum(1)
    condition = jnp.greater_equal(cost, 1.0)
    cost = jnp.where(condition, 1.0, 0.0)

    cost = cost.reshape(recon.shape[0], recon.shape[1])
    return cost



  def _metrics(self, data, dists, post, prior, losses, model_loss):
    entropy = lambda feat: self.rssm.get_dist(feat).entropy()
    metrics = {}
    metrics.update(jaxutils.tensorstats(entropy(prior), 'prior_ent'))
    metrics.update(jaxutils.tensorstats(entropy(post), 'post_ent'))
    metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    metrics['model_loss_mean'] = model_loss.mean()
    metrics['model_loss_std'] = model_loss.std()
    metrics['reward_max_data'] = jnp.abs(data['reward']).max()
    metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()
    if 'reward' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
      metrics.update({f'reward_{k}': v for k, v in stats.items()})
    if 'cont' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
      metrics.update({f'cont_{k}': v for k, v in stats.items()})
    if 'cost' in data.keys():
      metrics['cost_max_data'] = jnp.abs(data['cost']).max()
    if 'cost' in dists.keys():
      metrics['cost_max_pred'] = jnp.abs(dists['cost'].mean()).max()
    if 'cost' in dists and 'cost' in data.keys() and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['cost'], data['cost'], 0.1)
      metrics.update({f'cost_{k}': v for k, v in stats.items()})
    return metrics

class ImagSafeActorCritic(nj.Module):
  def __init__(self, critics, cost_critics, p_critics, scales, cost_scales, p_scales, act_space, config, online = True):
    critics = {k: v for k, v in critics.items() if scales[k]}
    cost_critics = {k: v for k, v in cost_critics.items() if scales[k]}
    p_critics = {k: v for k, v in p_critics.items() if p_scales[k]}

    for key, scale in scales.items():
      assert not scale or key in critics, key
    for key, cost_scale in cost_scales.items():
      assert not cost_scale or key in cost_critics, key
    self.critics = {k: v for k, v in critics.items() if scales[k]}
    self.cost_critics = {k: v for k, v in cost_critics.items() if cost_scales[k]}
    self.p_critics = {k: v for k, v in p_critics.items() if p_scales[k]}

    self.scales = scales
    self.cost_scales = cost_scales
    self.p_scales = p_scales
    self.act_space = act_space
    self.config = config
    self.lagrange = jaxutils.Lagrange(self.config.lagrange_multiplier_init, self.config.penalty_multiplier_init, self.config.cost_limit, name=f'lagrange')
    disc = act_space.discrete
    self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
    self.actor = nets.MLP(
        name='actor', dims='deter', shape=act_space.shape, **config.actor,
        dist=config.actor_dist_disc if disc else config.actor_dist_cont)
    self.retnorms = {
        k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
        for k in critics}
    self.costnorms = {
        k: jaxutils.Moments(**config.costnorm, name=f'costnorm_{k}')
        for k in cost_critics}
    self.pnorms = {
        k: jaxutils.Moments(**config.costnorm, name=f'pnorm_{k}')
        for k in p_critics}
    self.opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)
    self.online = online

    self.pre_actor = nets.MLP(
        name='actor',
        dims='deter',
        shape=act_space.shape,
        **config.actor,
        dist=config.actor_dist_disc if disc else config.actor_dist_cont
    )
    self.pre_opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)

  def initial(self, batch_size):
    return {}
  
  def init_actor_params(self, dummy_shape):
    dummy_input = jnp.zeros(dummy_shape)
    _ = self.actor(dummy_input)

  def policy(self, state, carry):
    if self.online == True:
      online_action = self.actor(state)
      offline_action = self.pre_actor(state)
      online_state = state
      offline_state = state
      online_state['action'] = online_action.sample(seed=nj.rng())
      offline_state['action'] = offline_action.sample(seed=nj.rng())
      for key, critic in self.critics.items():
        for k, cost_critic in self.cost_critics.items():
          q1 = critic.net(online_state).mean()
          q2 = critic.net(offline_state).mean()
          cost_q1 = cost_critic.net(online_state).mean()
          cost_q2 = cost_critic.net(offline_state).mean()
          q1 = q1 - 0.1*cost_q1
          q2 = q2 - 0.1*cost_q2
          q = jnp.stack([q1, q2], axis=-1)
          inv_temperature = 5
          logits = q * inv_temperature
          w_dist = Categorical(logits=logits)
          w = w_dist.sample(seed=nj.rng())
          def select_action_dist(w, online_action, offline_action):
            return jax.lax.cond(
                w > 0.5,
                lambda _: offline_action,
                lambda _: online_action,
                operand=None,
            )
          w_scalar = w[0]

          action = select_action_dist(w_scalar, online_action, offline_action)
      
      return {'action': action}, carry
    
    else:
      return {'action': self.pre_actor(state)}, carry

  def train(self, imagine, start, context):
    def loss(start):
      policy = lambda s: self.pre_actor(sg(s)).sample(seed=nj.rng())
      traj = imagine(policy, start, self.config.imag_horizon)
      loss, metrics = self.loss(traj)
      return loss, (traj, metrics)
    return_to_go = [0] * len(start['reward'])
    prev_return = 0
    for i in range(len(start['reward'])):
      return_to_go[-i-1] = start['reward'][-i-1] + (1 - 1 / self.config.horizon) * prev_return * (1 - start['is_last'][-i-1])
      prev_return = return_to_go[-i-1]
    return_to_go = jnp.stack(list(return_to_go))
    start['return'] = return_to_go
    rew = start['reward']
    last = start['is_last']
    action = start['action']
    ret = start['return']
    cost = start['cost']
    if self.online==True:
      mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
      metrics.update(mets)
      for key, critic in self.critics.items():
        mets = critic.train(traj, rew, last, action, ret, self.actor, self.online)
        metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
      for key, cost_critic in self.cost_critics.items():
        mets = cost_critic.train(traj, rew, last, action, cost, self.actor, self.online)
        metrics.update({f'{key}_cost_critic_{k}': v for k, v in mets.items()})
      for key, p_critic in self.p_critics.items():
        mets = p_critic.train(traj, self.actor)
        metrics.update({f'{key}_p_critic_{k}': v for k, v in mets.items()})
    else:
      mets, (traj, metrics) = self.pre_opt(self.pre_actor, loss, start, has_aux=True)
      metrics.update(mets)
      for key, critic in self.critics.items():
        mets = critic.train(traj, rew, last, action, ret, self.pre_actor, self.online)
        metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
      for key, cost_critic in self.cost_critics.items():
        mets = cost_critic.train(traj, rew, last, action, cost, self.pre_actor, self.online)
        metrics.update({f'{key}_cost_critic_{k}': v for k, v in mets.items()})
      for key, p_critic in self.p_critics.items():
        mets = p_critic.train(traj, self.pre_actor)
        metrics.update({f'{key}_p_critic_{k}': v for k, v in mets.items()})
    return traj, metrics


  def loss(self, traj):
    metrics = {}
    advs = []
    total = sum(self.scales[k] for k in self.critics)
    
    if self.online == True:
      for key, critic in self.critics.items():
        rew, ret, base, q = critic.score(traj, self.actor)
        offset, invscale = self.retnorms[key](q)
        normed_ret = (q - offset) / invscale
        normed_base = (base - offset) / invscale
        advs.append((normed_ret - normed_base) * self.scales[key] / total)
        metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
        metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
        metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
        metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
      adv = jnp.stack(advs).sum(0)
      
      cost_advs = []
      cost_total = sum(self.cost_scales[k] for k in self.cost_critics)
      cost_vs = []
      for key, cost_critic in self.cost_critics.items():
        cost, cost_ret, base, qc, vc = cost_critic.score(traj, self.actor)
        cost_vs.append(vc)
        offset, invscale = self.costnorms[key](qc)
        normed_ret = (qc - offset) / invscale
        normed_base = (base - offset) / invscale
        cost_advs.append((normed_base - normed_ret) * self.cost_scales[key] / total)
        metrics.update(jaxutils.tensorstats(cost, f'{key}_cost'))
        metrics.update(jaxutils.tensorstats(cost_ret, f'{key}_cost_raw'))
        metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_cost_normed'))
        metrics[f'{key}_cost_rate'] = (jnp.abs(cost_ret) >= 0.5).mean()
        if self.config.pessimistic: 
          cost_ret_episode = jnp.stack(cost_ret).sum(0)
        else:
          cost_ret_episode = jnp.stack(cost_ret).mean(0)
      
      p_vals = []
      for key, p_critic in self.p_critics.items():
        p_val, p_ret = p_critic.score(traj, self.actor)
        offset, invscale = self.pnorms[key](p_ret)
        normed_pval = (p_val - offset) / invscale
        p_vals.append(normed_pval)
      pvals = jnp.stack(p_vals).sum(0)
  
      unsafe_condition = pvals
      safe_condition = (1 - pvals) * jnp.where(qc<=0. , 1, 0)
      cost_adv = jnp.stack(cost_advs).sum(0)
      unsafe_weights = unsafe_condition * jnp.exp(sg(cost_adv)) ## ignore vc >0, qc>vc
      safe_weights = safe_condition * jnp.exp(sg(adv))
      weight = unsafe_weights + safe_weights
      
      policy = self.actor(sg(traj))
      logpi = policy.log_prob(sg(traj['action']))[:-1]
      loss = {'backprop': -adv, 'reinforce': -logpi * weight}[self.grad]
      ent = policy.entropy()[:-1]
      loss -= self.config.actent * ent
      loss *= sg(traj['weight'])[:-1]
      loss *= self.config.loss_scales.actor
      metrics.update(self._metrics(traj, policy, logpi, ent, adv))
      loss = loss.mean()

      if self.config.pessimistic: 
        cost_ret_episode = jnp.stack((1 - pvals)*cost_ret).sum(0)
      else:
        cost_ret_episode = jnp.stack((1 - pvals)*cost_ret).mean(0)
        
      penalty, lagrange_multiplier, penalty_multiplier = self.lagrange(cost_ret_episode)
      metrics[f'lagrange_multiplier'] = lagrange_multiplier
      metrics[f'penalty_multiplier'] = penalty_multiplier
      metrics[f'penalty'] = penalty
      loss += penalty
    else:
      for key, critic in self.critics.items():
        rew, ret, base, q = critic.score(traj, self.pre_actor)
        offset, invscale = self.retnorms[key](q)
        normed_ret = (q - offset) / invscale
        normed_base = (base - offset) / invscale
        advs.append((normed_ret - normed_base) * self.scales[key] / total)
        metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
        metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
        metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
        metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
      adv = jnp.stack(advs).sum(0)
      
      cost_advs = []
      cost_total = sum(self.cost_scales[k] for k in self.cost_critics)
      cost_vs = []
      for key, cost_critic in self.cost_critics.items():
        cost, cost_ret, base, qc, vc = cost_critic.score(traj, self.pre_actor)
        cost_vs.append(vc)
        offset, invscale = self.costnorms[key](qc)
        normed_ret = (qc - offset) / invscale
        normed_base = (base - offset) / invscale
        cost_advs.append((normed_base - normed_ret) * self.cost_scales[key] / total)
        metrics.update(jaxutils.tensorstats(cost, f'{key}_cost'))
        metrics.update(jaxutils.tensorstats(cost_ret, f'{key}_cost_raw'))
        metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_cost_normed'))
        metrics[f'{key}_cost_rate'] = (jnp.abs(cost_ret) >= 0.5).mean()
        if self.config.pessimistic: 
          cost_ret_episode = jnp.stack(cost_ret).sum(0)
        else:
          cost_ret_episode = jnp.stack(cost_ret).mean(0)

      p_vals = []
      for key, p_critic in self.p_critics.items():
        p_val, p_ret = p_critic.score(traj, self.pre_actor)
        offset, invscale = self.pnorms[key](p_ret)
        normed_pval = (p_val - offset) / invscale
        p_vals.append(normed_pval)
      pvals = jnp.stack(p_vals).sum(0)

      unsafe_condition = pvals
      safe_condition = (1 - pvals) * jnp.where(qc<=0. , 1, 0)
      cost_adv = jnp.stack(cost_advs).sum(0)
      unsafe_weights = unsafe_condition * jnp.exp(sg(cost_adv)) ## ignore vc >0, qc>vc
      safe_weights = safe_condition * jnp.exp(sg(adv))
      weight = unsafe_weights + safe_weights

      policy = self.pre_actor(sg(traj))
      logpi = policy.log_prob(sg(traj['action']))[:-1]
      loss = {'backprop': -adv, 'reinforce': -logpi * weight}[self.grad]
      ent = policy.entropy()[:-1]
      loss -= self.config.actent * ent
      loss *= sg(traj['weight'])[:-1]
      loss *= self.config.loss_scales.actor
      metrics.update(self._metrics(traj, policy, logpi, ent, adv))
      loss = loss.mean()

      if self.config.pessimistic: 
        cost_ret_episode = jnp.stack((1 - pvals)*cost_ret).sum(0)
      else:
        cost_ret_episode = jnp.stack((1 - pvals)*cost_ret).mean(0)
        
      penalty, lagrange_multiplier, penalty_multiplier = self.lagrange(cost_ret_episode)
      metrics[f'lagrange_multiplier'] = lagrange_multiplier
      metrics[f'penalty_multiplier'] = penalty_multiplier
      metrics[f'penalty'] = penalty
      loss += penalty
    return loss, metrics

  def _metrics(self, traj, policy, logpi, ent, adv):
    metrics = {}
    ent = policy.entropy()[:-1]
    rand = (ent - policy.minent) / (policy.maxent - policy.minent)
    rand = rand.mean(range(2, len(rand.shape)))
    act = traj['action']
    act = jnp.argmax(act, -1) if self.act_space.discrete else act
    metrics.update(jaxutils.tensorstats(act, 'action'))
    metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
    metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
    metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    return metrics

class VFunction(nj.Module):

  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    self.vnet = nets.MLP((), name='vnet', dims='deter', **self.config.value)
    self.net = nets.MLP((), name='net', dims='deter', **self.config.critic)
    self.slow = nets.MLP((), name='slow', dims='deter', **self.config.critic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.vopt = jaxutils.Optimizer(name='critic_vopt', **self.config.value_opt)
    self.opt = jaxutils.Optimizer(name='critic_opt', **self.config.critic_opt)

  def train(self, traj, rew, last, action, ret, actor, online):
    target = sg(self.score(traj)[1])
    traj_start = {key: value[0, :, ...] if isinstance(value, jnp.ndarray) else value for key, value in traj.items()}
    traj_start['reward'] = rew
    traj_start['is_last'] =  last
    traj_start['action'] = action
    target_offline = sg(self.reg_score(traj_start, self.score(traj)[1])[1])
    vmets, vmetrics = self.vopt(self.vnet, self.vloss, traj, has_aux=True)
    mets, metrics = self.opt(self.net, self.loss, traj, traj_start, target, target_offline, online=online, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics
  
  def vloss(self, traj):
    def expect(diff, loss, expectile=0.8):
      weight = jnp.where(diff > 0, expectile, (1 - expectile))
      return weight * loss
    
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    vdist = self.vnet(traj)
    qdist = self.net(traj)
    qslowdist = self.slow(traj)
    diff = vdist.mean() - qdist.mean()
    loss_ori = -vdist.log_prob(sg(qdist.mean()))
    loss = expect(diff, loss_ori)
    reg_ori = -vdist.log_prob(sg(qslowdist.mean()))
    diff_reg = vdist.mean() - qslowdist.mean()
    reg = expect(diff_reg, reg_ori)
    # loss += self.config.loss_scales.slowreg * reg # do not use regulation
    loss = loss.mean()
    loss *= self.config.loss_scales.critic
    return loss, metrics

  def loss(self, traj, traj_reg, target, target_reg, online=False):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(self.slow(traj).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    loss = (loss * sg(traj['weight'])).mean()
    loss *= self.config.loss_scales.critic
    if online==False:
      traj_reg = {k: v[:-1] for k, v in traj_reg.items()}
      dist_reg = self.net(traj_reg)
      reg = -dist_reg.log_prob(sg(self.slow(traj_reg).mean()))
      loss_off = self.config.loss_scales.critic * ((-dist_reg.log_prob(sg(target_reg)) + self.config.loss_scales.slowreg * reg) * sg(traj_reg['weight'])).mean()
      loss += loss_off
      metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, actor=None):
    rew = self.rewfn(traj)
    assert len(rew) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    value = self.vnet(traj).mean()
    vals = [value[-1]]
    interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    qval = self.net(traj).mean()
    return rew, ret, value[:-1], qval[:-1]
  
  def reg_score(self, traj, ret, actor=None):
    ret = ret[0][1:]
    rew = traj['reward'][:-1]
    discount = 1 - 1 / self.config.horizon
    value = self.vnet(traj).mean()
    vals = []
    for t in range(len(rew)):
      vals.append(rew[t] + (1 - traj['is_last'][t]) * discount * ret[t])
    ret = jnp.stack(list(vals))
    return rew, ret, value[:-1]

class CostVFunction(nj.Module):

  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    self.vnet = nets.MLP((), name='vnet', dims='deter', **self.config.value)
    self.net = nets.MLP((), name='net', dims='deter', **self.config.critic)
    self.slow = nets.MLP((), name='slow', dims='deter', **self.config.critic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.vopt = jaxutils.Optimizer(name='cost_critic_vopt', **self.config.value_opt)
    self.opt = jaxutils.Optimizer(name='cost_critic_opt', **self.config.critic_opt)

  def train(self, traj, rew, last, action, cost, actor, online):
    target = sg(self.score(traj)[1])
    traj_start = {key: value[0, :, ...] if isinstance(value, jnp.ndarray) else value for key, value in traj.items()}
    traj_start['reward'] = rew
    traj_start['is_last'] =  last
    traj_start['action'] = action
    traj_start['cost'] = cost
    target_offline = sg(self.reg_score(traj_start, self.score(traj)[1])[1])
    vmets, vmetrics = self.vopt(self.vnet, self.vloss, traj, has_aux=True)
    mets, metrics = self.opt(self.net, self.loss, traj, traj_start, target, target_offline, online=online, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics
  
  def vloss(self, traj):
    def expect(diff, loss, expectile=0.8):
      weight = jnp.where(diff > 0, expectile, (1 - expectile))
      return weight * loss
    
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    vdist = self.vnet(traj)
    qdist = self.net(traj)
    qslowdist = self.slow(traj)
    diff = vdist.mean() - qdist.mean()
    loss_ori = -vdist.log_prob(sg(qdist.mean()))
    loss = expect(diff, loss_ori)
    reg_ori = -vdist.log_prob(sg(qslowdist.mean()))
    diff_reg = vdist.mean() - qslowdist.mean()
    reg = expect(diff_reg, reg_ori)
    # loss += self.config.loss_scales.slowreg * reg # do not use regulation
    loss = loss.mean()
    loss *= self.config.loss_scales.critic
    return loss, metrics

  def loss(self, traj, traj_reg, target, target_reg, online=False):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(self.slow(traj).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    loss = (loss * sg(traj['weight'])).mean()
    loss *= self.config.loss_scales.critic
    if online == False:
      traj_reg = {k: v[:-1] for k, v in traj_reg.items()}
      dist_reg = self.net(traj_reg)
      reg = -dist_reg.log_prob(sg(self.slow(traj_reg).mean()))
      loss_off = self.config.loss_scales.critic * ((-dist_reg.log_prob(sg(target_reg)) + self.config.loss_scales.slowreg * reg) * sg(traj_reg['weight'])).mean()
      loss += loss_off
      metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, actor=None, lag=1.0):
    rew = self.rewfn(traj)
    rew_repeat = rew
    assert len(rew_repeat) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    value = self.vnet(traj).mean()
    vals = [lag * value[-1]]
    interm = rew_repeat + lag * disc * value[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    qval = self.net(traj).mean()
    return rew_repeat, ret, value[:-1], qval[:-1], value
  
  def reg_score(self, traj, ret, actor=None):
    ret = ret[0][1:]
    rew = traj['cost'][:-1]
    discount = 1 - 1 / self.config.horizon
    value = self.vnet(traj).mean()
    vals = []
    for t in range(len(rew)):
      vals.append(rew[t] + (1 - traj['is_last'][t]) * discount * ret[t])
    ret = jnp.stack(list(vals))
    return rew, ret, value[:-1]
  
class RefVFunction(nj.Module):

  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), name='net', dims='deter', **self.config.critic)
    self.slow = nets.MLP((), name='slow', dims='deter', **self.config.critic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.opt = jaxutils.Optimizer(name='pcritic_opt', **self.config.cost_critic_opt)

  def train(self, traj, actor):
    target = sg(self.score(traj)[1])
    mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics

  def loss(self, traj, target):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(self.slow(traj).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    loss = (loss * sg(traj['weight'])).mean()
    loss *= self.config.loss_scales.critic
    metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, actor=None):
    cost_v = self.net(traj).mean()
    unsafe = jnp.where( cost_v > 0. , 1, 0)
    pret_buf = unsafe.copy()
    mu = 0.99
    pret = [pret_buf[-1]]
    for i in reversed(range(pret_buf.shape[0]-1)):
      pret.append(jnp.maximum(pret_buf[i], mu * pret[-1]))
    pret = jnp.stack(list(reversed(pret))[:-1])
    return cost_v[:-1], pret