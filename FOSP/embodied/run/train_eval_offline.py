import re

import embodied
import numpy as np

class CostEma:

  def __init__(self, initial=0):
    self.value = initial

class Arrive:

  def __init__(self):
    self.value = []

def train_eval_offline(
    agent, train_env, eval_env, train_replay, eval_replay, logger, args, lag):
  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_eval = embodied.when.Every(args.eval_every, args.eval_initial)
  should_sync = embodied.when.Every(args.sync_every)
  step = logger.step
  cost_ema = CostEma(0.0)
  train_arrive_num = Arrive()
  eval_arrive_num = Arrive()
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  print('Observation space:', embodied.format(train_env.obs_space), sep='\n')
  print('Action space:', embodied.format(train_env.act_space), sep='\n')

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', train_env, ['step'])
  if hasattr(train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])

  nonzeros = set()
  def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    logger.add({
        'length': length,
        'score': score,
    }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
    print(f'Episode has {length} steps and return {score:.1f}.')
    if 'cost' in ep.keys():
      cost = float(ep['cost'].astype(np.float64).sum())
      logger.add({
          'cost': cost,
      }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
      print(f'Episode has {length} steps and  cost {cost:.1f}.')
      # lag.add_cost(cost)
      cost_ema.value = cost_ema.value * 0.99 + cost * 0.01
      logger.add({
          'cost_ema': cost_ema.value,
      }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
      if step > 5000:
        lag.pid_update(cost_ema.value, step)
    if 'arrive_dest' in ep.keys():
      if mode == 'train':
        train_arrive_num.value.append(int(ep['arrive_dest'][-1]))
        if len(train_arrive_num.value) == 10:
          arrive_rate = sum(train_arrive_num.value) /  10
          train_arrive_num.value = []
          logger.add({
              'arrive_rate': arrive_rate,
          }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
          print(f'train 10 episodes has average arrive rate {arrive_rate:.2f}.')

      else:
        eval_arrive_num.value.append(int(ep['arrive_dest'][-1]))
        if len(eval_arrive_num.value) == 10:
          arrive_rate = sum(eval_arrive_num.value) /  10
          eval_arrive_num.value = []
          logger.add({
              'arrive_rate': arrive_rate,
          }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
          print(f'eval 10 episodes has average arrive rate {arrive_rate:.2f}.')


    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
    metrics.add(stats, prefix=f'{mode}_stats')

  driver_train = embodied.Driver(train_env)
  driver_train.on_episode(lambda ep, worker: per_episode(ep, mode='train'))
  driver_train.on_step(lambda tran, _: step.increment())
  driver_eval = embodied.Driver(eval_env)
  driver_eval.on_step(eval_replay.add)
  driver_eval.on_episode(lambda ep, worker: per_episode(ep, mode='eval'))

  # Prefill train dataset
  random_agent = embodied.RandomAgent(train_env.act_space)
  print('Prefill eval dataset.')
  while len(eval_replay) < max(args.batch_steps, args.eval_fill):
    driver_eval(random_agent.policy, steps=100, lag=lag.lagrange_penalty, lag_p=lag.delta_p, lag_i=lag.pid_i, lag_d=lag.pid_d)
  logger.add(metrics.result())
  logger.write()
  
  # Load the dataset
  print('Load the dataset.')
  train_replay.load(load_dir = True)
  print(train_replay.stats)

  dataset_train = agent.dataset(train_replay.dataset)
  dataset_eval = agent.dataset(eval_replay.dataset)
  state = [None]  # To be writable from train step function below.
  batch = [None]
  def train_step(tran, worker):
    for _ in range(should_train(step)):
      with timer.scope('dataset_train'):
        batch[0] = next(dataset_train)
      outs, state[0], mets = agent.train(batch[0], state[0])
      metrics.add(mets, prefix='train')
      if 'priority' in outs:
        train_replay.prioritize(outs['key'], outs['priority'])
      updates.increment()
    if should_sync(updates):
      agent.sync()
    if should_log(step):
      logger.add(metrics.result())
      logger.add(agent.report(batch[0]), prefix='report')
      with timer.scope('dataset_eval'):
        eval_batch = next(dataset_eval)
      logger.add(agent.report(eval_batch), prefix='eval')
      logger.add(train_replay.stats, prefix='replay')
      logger.add(eval_replay.stats, prefix='eval_replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  driver_train.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we jused saved.

  print('Start training loop.')
  policy_train = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  policy_eval = lambda *args: agent.policy(*args, mode='eval')
  while step < args.steps:
    if should_eval(step):
      print('Starting evaluation at step', int(step))
      driver_eval.reset()
      driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps), lag=lag.lagrange_penalty, lag_p=lag.delta_p, lag_i=lag.pid_i, lag_d=lag.pid_d)
    driver_train(policy_train, steps=100, lag=lag.lagrange_penalty, lag_p=lag.delta_p, lag_i=lag.pid_i, lag_d=lag.pid_d)
    if should_save(step):
      checkpoint.save()
  logger.write()
  logger.write()

