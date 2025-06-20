import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from . import agent
from . import agent
from . import expl
from . import ninjax as nj
from . import jaxutils

import jax
from jax import lax

tree_map = jax.tree_util.tree_map

sg = lambda x: tree_map(jax.lax.stop_gradient, x)

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

class Greedy(nj.Module):

  def __init__(self, wm, act_space, obs_space, step, act_space2, config):
    rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
    if config.use_cost_model:
      costfn = lambda s: wm.heads['cost'](s).mean()[1:]
    else:
      costfn = lambda s: cost_from_state(wm, s)[1:]

    critics = {'extr': agent.VFunction(rewfn, config, name='critic')}
    cost_critics = {'extr': agent.CostVFunction(costfn, config, name='cost_critic')}
    p_critics = {'extr': agent.RefVFunction(costfn, config, name='p_critic')}

    if config.use_pex:
      self.ac = agent.ImagSafeActorCritic(
          critics, cost_critics, p_critics, {'extr': 1.0}, {'extr': 1.0}, {'extr': 1.0}, act_space, config, online=True, name='safe_ac')
    else:
      self.ac = agent.ImagSafeActorCritic(
          critics, cost_critics, p_critics, {'extr': 1.0}, {'extr': 1.0}, {'extr': 1.0}, act_space, config, online=False, name='safe_ac')

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    return self.ac.policy(latent, state)

  def train(self, imagine, start, data):
    return self.ac.train(imagine, start, data)

  def report(self, data):
    return {}

class Random(nj.Module):

  def __init__(self, wm, act_space, config):
    self.config = config
    self.act_space = act_space

  def initial(self, batch_size):
    return jnp.zeros(batch_size)

  def policy(self, latent, state):
    batch_size = len(state)
    shape = (batch_size,) + self.act_space.shape
    if self.act_space.discrete:
      dist = jaxutils.OneHotDist(jnp.zeros(shape))
    else:
      dist = tfd.Uniform(-jnp.ones(shape), jnp.ones(shape))
      dist = tfd.Independent(dist, 1)
    return {'action': dist}, state

  def train(self, imagine, start, data):
    return None, {}

  def report(self, data):
    return {}


class Explore(nj.Module):

  REWARDS = {
      'disag': expl.Disag,
  }

  def __init__(self, wm, act_space, config):
    self.config = config
    self.rewards = {}
    critics = {}
    for key, scale in config.expl_rewards.items():
      if not scale:
        continue
      if key == 'extr':
        rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
        critics[key] = agent.VFunction(rewfn, config, name=key)
      else:
        rewfn = self.REWARDS[key](
            wm, act_space, config, name=key + '_reward')
        critics[key] = agent.VFunction(rewfn, config, name=key)
        self.rewards[key] = rewfn
    scales = {k: v for k, v in config.expl_rewards.items() if v}
    self.ac = agent.ImagActorCritic(
        critics, scales, act_space, config, name='ac')

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    return self.ac.policy(latent, state)

  def train(self, imagine, start, data):
    metrics = {}
    for key, rewfn in self.rewards.items():
      mets = rewfn.train(data)
      metrics.update({f'{key}_k': v for k, v in mets.items()})
    traj, mets = self.ac.train(imagine, start, data)
    metrics.update(mets)
    return traj, metrics

  def report(self, data):
    return {}