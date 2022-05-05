from collections import OrderedDict
import time, os, keyboard # , talib, bottleneck
import multiprocessing as mp
curdir = os.path.expanduser("~")
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0,1,2,3
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit' # lets XLA work on CPU
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
# tf.config.run_functions_eagerly(True)
# tf.config.optimizer.set_jit("autoclustering") # enable XLA
# tf.config.experimental.enable_mlir_graph_optimization()
# tf.random.set_seed(0) # TODO https://www.tensorflow.org/guide/random_numbers
tf.keras.backend.set_epsilon(tf.experimental.numpy.finfo(tf.keras.backend.floatx()).eps) # 1e-7 default
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import gym, gym_algorithmic, procgen, pybullet_envs

# CUDA 11.2.2_461.33, CUDNN 8.1.1.33, tensorflow-gpu==2.6.0, tensorflow_probability==0.14.0
physical_devices_gpu = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices_gpu)): tf.config.experimental.set_memory_growth(physical_devices_gpu[i], True)
import gym_util, model_util as util, model_nets as nets

# TODO add Fourier prior like PercieverIO or https://github.com/zongyi-li/fourier_neural_operator
# TODO add S4 layer https://github.com/HazyResearch/state-spaces
# TODO how does CLIP quantize latents? https://github.com/openai/CLIP

# TODO try out MuZero-ish architecture
# TODO add Perciever, maybe ReZero

# TODO add GenNet and DisNet for GAN type boost
# TODO put actor in seperate process so can run async
# TODO add ZMQ and latent pooling

# TODO how to incorporate ARS random policy search?
# TODO try out the 'lottery ticket hypothosis' pruning during training
# TODO use numba to make things faster on CPU


class GeneralAI(tf.keras.Model):
    def __init__(self, arch, env, trader, env_render, max_episodes, max_steps, learn_rate, entropy_contrib, returns_disc, value_cont, force_cont_obs, force_cont_action, latent_size, latent_dist, net_attn_io, aio_max_latents, attn_mem_multi, aug_data_step, aug_data_pos):
        super(GeneralAI, self).__init__()
        compute_dtype = tf.dtypes.as_dtype(self.compute_dtype)
        self.float_min = tf.constant(compute_dtype.min, compute_dtype)
        self.float_maxroot = tf.constant(tf.math.sqrt(compute_dtype.max), compute_dtype)
        self.float_eps = tf.constant(tf.experimental.numpy.finfo(compute_dtype).eps, compute_dtype)
        self.float64_eps = tf.constant(tf.experimental.numpy.finfo(tf.float64).eps, tf.float64)
        # self.float_log_min_prob = tf.constant(tf.math.log(self.float_eps), compute_dtype)
        self.compute_zero, self.int32_max, self.int32_maxbit, self.int32_zero, self.float64_zero = tf.constant(0, compute_dtype), tf.constant(tf.int32.max, tf.int32), tf.constant(1073741824, tf.int32), tf.constant(0, tf.int32), tf.constant(0, tf.float64)

        self.arch, self.env, self.trader, self.env_render, self.value_cont, self.force_cont_obs, self.force_cont_action = arch, env, trader, env_render, value_cont, force_cont_obs, force_cont_action
        self.max_episodes, self.max_steps, self.learn_rate, self.attn_mem_multi, self.entropy_contrib, self.returns_disc = tf.constant(max_episodes, tf.int32), tf.constant(max_steps, tf.int32), tf.constant(learn_rate, tf.float64), tf.constant(attn_mem_multi, tf.int32), tf.constant(entropy_contrib, compute_dtype), tf.constant(returns_disc, tf.float64)
        self.dist_prior = tfp.distributions.Independent(tfp.distributions.Logistic(loc=tf.zeros(latent_size, dtype=self.compute_dtype), scale=10.0), reinterpreted_batch_ndims=1)
        # self.dist_prior = tfp.distributions.Independent(tfp.distributions.Uniform(low=tf.cast(tf.fill(latent_size,-10), dtype=self.compute_dtype), high=10), reinterpreted_batch_ndims=1)
        self.initializer = tf.keras.initializers.GlorotUniform()

        self.obs_spec, self.obs_zero, _ = gym_util.get_spec(env.observation_space, self.compute_dtype, force_cont=force_cont_obs)
        self.action_spec, _, self.action_zero_out = gym_util.get_spec(env.action_space, self.compute_dtype, force_cont=force_cont_action)
        self.obs_spec_len, self.action_spec_len = len(self.obs_spec), len(self.action_spec)
        self.gym_step_shapes = [feat['step_shape'] for feat in self.obs_spec] + [tf.TensorShape((1,1)), tf.TensorShape((1,1))]
        self.gym_step_dtypes = [feat['dtype'] for feat in self.obs_spec] + [tf.float64, tf.bool]
        self.rewards_zero, self.dones_zero = tf.constant([[0]],tf.float64), tf.constant([[False]],tf.bool)

        self.attn_img_scales = int(np.log(max_steps) / np.log(attn_mem_multi)) # TODO remove
        self.attn_img_step_sizes = [None]*(self.attn_img_scales)
        for step_scale in range(self.attn_img_scales): self.attn_img_step_sizes[step_scale] = tf.math.pow(self.attn_mem_multi, step_scale+1)
        # self.attn_img_step_sizes[-1] = tf.math.pow(self.attn_mem_multi, int(np.log2(max_steps))) # TODO remove
        self.attn_img_step_sizesT = tf.concat(self.attn_img_step_sizes, axis=0)
        if self.attn_img_step_sizesT.shape == (): self.attn_img_step_sizesT = tf.reshape(self.attn_img_step_sizesT, (1,))
        self.attn_img_step_locs = max_steps - tf.cast(max_steps / self.attn_img_step_sizesT, tf.int32)

        net_attn, net_lstm = True, False

        lat_batch_size = 0
        for i in range(self.obs_spec_len):
            if net_attn_io:
                event_size = self.obs_spec[i]['event_size']
                num_latents = aio_max_latents if event_size > aio_max_latents else event_size
            else: num_latents = 1
            self.obs_spec[i]['num_latents'] = num_latents
            lat_batch_size += num_latents
        lat_batch_size += (1 if aug_data_step else 0)
        net_attn_io2 = (net_attn_io and lat_batch_size > aio_max_latents)
        if net_attn_io2: lat_batch_size = aio_max_latents
        memory_size = lat_batch_size * max_steps

        lat_batch_size_trans = lat_batch_size
        for i in range(self.action_spec_len):
            if net_attn_io:
                event_size = self.action_spec[i]['event_size']
                num_latents = aio_max_latents if event_size > aio_max_latents else event_size
            else: num_latents = 1
            self.action_spec[i]['num_latents'] = num_latents
            lat_batch_size_trans += num_latents
        lat_batch_size_trans += 1 # step_size
        memory_size_trans = lat_batch_size_trans * max_steps

        latent_spec = {'dtype':compute_dtype, 'num_latents':lat_batch_size, 'num_latents_trans':lat_batch_size_trans, 'step_shape':tf.TensorShape((lat_batch_size,latent_size))}
        if latent_dist == 0: latent_spec.update({'event_shape':(latent_size,), 'num_components':0}) # deterministic
        if latent_dist == 1: latent_spec.update({'event_shape':(latent_size, latent_size), 'num_components':0}) # categorical
        if latent_dist == 2: latent_spec.update({'event_shape':(latent_size,), 'num_components':int(latent_size/16)}) # continuous
        self.latent_spec = latent_spec

        inputs = {'obs':tf.constant([[0,0,0]],compute_dtype)}
        opt_spec = [{'name':'meta', 'type':'ar', 'schedule_type':'', 'learn_rate':tf.constant(2e-5, tf.float64), 'float_eps':self.float_eps}]
        self.meta_spec = [{'net_type':0, 'dtype':tf.float64, 'dtype_out':compute_dtype, 'min':self.float_eps, 'max':self.learn_rate, 'is_discrete':False, 'num_components':8, 'event_shape':(1,), 'step_shape':tf.TensorShape((1,1))}]
        self.meta = nets.GenNet('M', inputs, opt_spec, self.meta_spec, False, latent_size, net_blocks=2, net_attn=False, net_lstm=False, net_attn_io=False, force_det_out=False); outputs = self.meta(inputs)
        self.meta.optimizer_weights = util.optimizer_build(self.meta.optimizer['meta'], self.meta.trainable_variables)
        util.net_build(self.meta, self.initializer)

        inputs = {'obs':self.obs_zero, 'rewards':self.rewards_zero, 'dones':self.dones_zero, 'step_size':1}
        if arch in ('PG','AC','TRANS','MU','VPN','SPR','MU2','MU3',):
            opt_spec = [
                {'name':'act', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps},
                {'name':'PG', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps},
                {'name':'PGL', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps},
                {'name':'trans', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps},
                {'name':'rwd', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps},
                {'name':'done', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps},
            ]
            self.rep = nets.RepNet('RN', inputs, opt_spec, self.obs_spec, latent_spec, latent_dist, latent_size, net_blocks=0, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, net_attn_io2=net_attn_io2, num_heads=4, memory_size=memory_size, aug_data_step=aug_data_step, aug_data_pos=aug_data_pos)
            outputs = self.rep(inputs); rep_dist = self.rep.dist(outputs)
            self.latent_zero = tf.zeros_like(rep_dist.sample(), latent_spec['dtype'])
            inputs['obs'] = self.latent_zero
            self.rep.optimizer_weights = []
            for spec in opt_spec: self.rep.optimizer_weights += util.optimizer_build(self.rep.optimizer[spec['name']], self.rep.trainable_variables)
            util.net_build(self.rep, self.initializer)

        opt_spec = [{'name':'action', 'type':'ar', 'schedule_type':'', 'learn_rate':tf.constant(6e-4, tf.float64), 'float_eps':self.float_eps}]
        self.action = nets.GenNet('AN', inputs, opt_spec, self.action_spec, force_cont_action, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=memory_size, max_steps=max_steps, force_det_out=False); outputs = self.action(inputs)
        self.action.optimizer_weights = util.optimizer_build(self.action.optimizer['action'], self.action.trainable_variables)
        util.net_build(self.action, self.initializer)

        if arch in ('AC','MU','VPN','MU2',):
            opt_spec = [{'name':'value', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps}]
            if value_cont:
                value_spec = [{'net_type':0, 'dtype':compute_dtype, 'dtype_out':compute_dtype, 'is_discrete':False, 'num_components':8, 'event_shape':(1,), 'step_shape':tf.TensorShape((1,1))}]
                self.value = nets.GenNet('VN', inputs, opt_spec, value_spec, False, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=memory_size, max_steps=max_steps, force_det_out=False); outputs = self.value(inputs)
            else: self.value = nets.ValueNet('VN', inputs, opt_spec, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, num_heads=4, memory_size=memory_size); outputs = self.value(inputs)
            self.value.optimizer_weights = util.optimizer_build(self.value.optimizer['value'], self.value.trainable_variables)
            util.net_build(self.value, self.initializer)

        if arch in ('TRANS','MU','VPN','SPR','MU2','MU3',):
            inputs['actions'] = self.action_zero_out
            opt_spec = [{'name':'trans', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps}]
            latent_dist = 2; latent_spec = {'dtype':compute_dtype, 'num_latents':lat_batch_size, 'num_latents_trans':lat_batch_size_trans, 'event_shape':(latent_size,), 'num_components':8}
            self.trans = nets.TransNet('TN', inputs, opt_spec, self.action_spec, latent_spec, latent_dist, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=memory_size_trans); outputs = self.trans(inputs)
            self.trans.optimizer_weights = util.optimizer_build(self.trans.optimizer['trans'], self.trans.trainable_variables)
            util.net_build(self.trans, self.initializer)
        if arch in ('MU','MU2','MU3',):
            opt_spec = [{'name':'rwd', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps}]
            reward_spec = [{'net_type':0, 'dtype':tf.float64, 'dtype_out':compute_dtype, 'is_discrete':False, 'num_components':16, 'event_shape':(1,), 'step_shape':tf.TensorShape((1,1))}]
            self.rwd = nets.GenNet('RW', inputs, opt_spec, reward_spec, False, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=memory_size, max_steps=max_steps, force_det_out=False); outputs = self.rwd(inputs)
            self.rwd.optimizer_weights = util.optimizer_build(self.rwd.optimizer['rwd'], self.rwd.trainable_variables)
            util.net_build(self.rwd, self.initializer)
            opt_spec = [{'name':'done', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps}]
            done_spec = [{'net_type':0, 'dtype':tf.bool, 'dtype_out':tf.int32, 'is_discrete':True, 'num_components':2, 'event_shape':(1,), 'step_shape':tf.TensorShape((1,1))}]
            self.done = nets.GenNet('DO', inputs, opt_spec, done_spec, False, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=memory_size, max_steps=max_steps, force_det_out=False); outputs = self.done(inputs)
            self.done.optimizer_weights = util.optimizer_build(self.done.optimizer['done'], self.done.trainable_variables)
            util.net_build(self.done, self.initializer)



        metrics_loss = OrderedDict()
        metrics_loss['1rewards*'] = {'-rewards_ma':np.float64, '-rewards_total+':np.float64, 'rewards_final=':np.float64}
        metrics_loss['1steps'] = {'steps+':np.int64}
        if arch == 'PG':
            metrics_loss['1nets'] = {'loss_action':np.float64}
            metrics_loss['1extras'] = {'returns':np.float64}
        if arch == 'AC':
            metrics_loss['1nets'] = {'loss_action':np.float64, 'loss_value':np.float64}
            metrics_loss['1extras*'] = {'returns':np.float64, 'advantages':np.float64}
        if arch == 'TRANS':
            metrics_loss['1nets'] = {'loss_action':np.float64}
        if arch == 'MU':
            metrics_loss['1nets'] = {'loss_action':np.float64, 'loss_value':np.float64}
            metrics_loss['1nets1'] = {'loss_policy':np.float64, 'loss_return':np.float64}
            metrics_loss['1nets2'] = {'loss_rwd':np.float64, 'loss_done':np.float64}
            metrics_loss['1extras*'] = {'returns':np.float64, 'advantages':np.float64}
            # metrics_loss['nets3'] = {'loss_total_img':np.float64,'returns_img':np.float64}
            # metrics_loss['extras1'] = {'steps_img':np.float64}
        if arch == 'VPN':
            metrics_loss['1extra'] = {'returns_pred':np.float64}
            metrics_loss['1nets'] = {'actor_loss_action':np.float64}
            metrics_loss['1extra2'] = {'return_entropy':np.float64}
            metrics_loss['1nets1'] = {'loss_return':np.float64}
        if arch == 'SPR':
            metrics_loss['1nets'] = {'loss_action':np.float64}
            metrics_loss['1nets5'] = {'loss_next_action':np.float64}
            metrics_loss['1nets6'] = {'loss_next_action_img':np.float64}
        if arch == 'MU2':
            metrics_loss['1nets0'] = {'loss_action_planning':np.float64}
            # metrics_loss['1nets'] = {'loss_action':np.float64}
            metrics_loss['1nets2'] = {'loss_reward':np.float64, 'loss_reward_img':np.float64}
            metrics_loss['1nets3'] = {'loss_done':np.float64, 'loss_done_img':np.float64}
            metrics_loss['1nets4'] = {'loss_return':np.float64, 'loss_return_img':np.float64}
            metrics_loss['1nets5'] = {'loss_next_action':np.float64, 'loss_next_action_img':np.float64}
            # metrics_loss['1nets2'] = {'loss_reward':np.float64}
            # metrics_loss['1nets3'] = {'loss_done':np.float64}
            # metrics_loss['1nets4'] = {'loss_return':np.float64}
            # metrics_loss['1nets5'] = {'loss_next_action':np.float64}
        if arch == 'MU3':
            metrics_loss['1extra'] = {'returns_pred':np.float64}
            metrics_loss['1nets'] = {'actor_loss_action':np.float64}
            metrics_loss['1nets2'] = {'loss_rwd':np.float64, 'loss_done':np.float64}
            metrics_loss['1nets3'] = {'loss_rwd_dyn':np.float64, 'loss_done_dyn':np.float64}
            metrics_loss['1extra2'] = {'return_entropy':np.float64}
        if trader:
            metrics_loss['2trader_bal*'] = {'balance_avg':np.float64, 'balance_final=':np.float64}
            metrics_loss['1trader_marg*'] = {'equity':np.float64, 'margin_free':np.float64}
            metrics_loss['1trader_sim_time'] = {'sim_time_secs':np.float64}

        for loss_group in metrics_loss.values():
            for k in loss_group.keys():
                if k.endswith('=') or k.endswith('+'): loss_group[k] = [0 for i in range(max_episodes)]
                else: loss_group[k] = [[] for i in range(max_episodes)]
        self.metrics_loss = metrics_loss

        # TF bug that wont set graph options with tf.function decorator inside a class
        self.reset_states = tf.function(self.reset_states, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        self.reset_states()
        arch_run = getattr(self, arch); arch_run = tf.function(arch_run, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS); setattr(self, arch, arch_run)


    def metrics_update(self, *args):
        args = list(args)
        for i in range(1,len(args)): args[i] = args[i].item()
        log_metrics, episode, idx = args[0], args[1], 2
        for loss_group in self.metrics_loss.values():
            for k in loss_group.keys():
                if log_metrics[idx-2]:
                    if k.endswith('='): loss_group[k][episode] = args[idx]
                    elif k.endswith('+'): loss_group[k][episode] += args[idx]
                    else: loss_group[k][episode] += [args[idx]]
                idx += 1
        return np.asarray(0, np.int32) # dummy

    def env_reset(self, dummy):
        obs, reward, done = self.env.reset(), 0.0, False
        if self.env_render: self.env.render()
        if hasattr(self.env,'np_struc'): rtn = gym_util.struc_to_feat(obs)
        else: rtn = gym_util.space_to_feat(obs, self.env.observation_space)
        rtn += [np.asarray([[reward]], np.float64), np.asarray([[done]], bool)]
        return rtn
    def env_step(self, *args): # args = tuple of ndarrays
        if hasattr(self.env,'np_struc'): action = gym_util.out_to_struc(list(args), self.env.action_dtype)
        else: action = gym_util.out_to_space(args, self.env.action_space, [0])
        obs, reward, done, _ = self.env.step(action)
        if self.env_render: self.env.render()
        if hasattr(self.env,'np_struc'): rtn = gym_util.struc_to_feat(obs)
        else: rtn = gym_util.space_to_feat(obs, self.env.observation_space)
        rtn += [np.asarray([[reward]], np.float64), np.asarray([[done]], bool)]
        return rtn

    def check_stop(self, *args):
        if keyboard.is_pressed('ctrl+alt+k'): return np.asarray(True, bool)
        return np.asarray(False, bool)

    # TODO use ZMQ for remote messaging, latent pooling
    def transact_latents(self, *args):
        return [np.asarray([0,1,2], np.float64), np.asarray([2,1,0], np.float64)]


    def reset_states(self, use_img=False):
        for net in self.layers:
            if hasattr(net, 'reset_states'): net.reset_states(use_img=use_img)



    def VPN_actor(self, inputs):
        print("tracing -> GeneralAI VPN_actor")
        loss = {}
        loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        metric_entropy = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        metric_returns_pred = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        obs, actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): obs[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['event_shape'])
        for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
        rewards = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        dones = tf.TensorArray(tf.bool, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        returns = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))

        step = tf.constant(0)
        # while step < self.max_steps and not inputs['dones'][-1][0]:
        while not inputs['dones'][-1][0]:
            inputs_step = {}
            for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])

            with tf.GradientTape(persistent=True) as tape_action:
                rep_logits = self.rep(inputs, step=step); rep_dist = self.rep.dist(rep_logits)
                inputs_step['obs'] = rep_dist.sample()

            # outputs_img = self.VPN_imagine(inputs_step)
            # loss_img = self.VPN_img_learner(outputs_img)
            # action_rnd = [None]*self.action_spec_len
            # for i in range(self.action_spec_len):
            #     action_rnd[i] = tf.random.uniform((self.action_spec[i]['step_shape']), minval=self.action_spec[i]['min'], maxval=self.action_spec[i]['max'], dtype=self.action_spec[i]['dtype_out'])

            with tape_action:
                action_logits = self.action(inputs_step)
                action_dist, action, action_dis = [None]*self.action_spec_len, [None]*self.action_spec_len, [None]*self.action_spec_len
                for i in range(self.action_spec_len):
                    action_dist[i] = self.action.dist[i](action_logits[i])
                    action[i] = action_dist[i].sample()
                    actions[i] = actions[i].write(step, action[i][-1])
                    action_dis[i] = util.discretize(action[i], self.action_spec[i], self.force_cont_action)

            np_in = tf.numpy_function(self.env_step, action_dis, self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs['obs'], inputs['rewards'], inputs['dones'] = np_in[:-2], np_in[-2], np_in[-1]

            entropy = tf.constant([0.0], dtype=self.compute_dtype)
            inputs_step['actions'] = action
            trans_logits = self.trans(inputs_step); trans_dist = self.trans.dist(trans_logits)
            inputs_step['obs'] = trans_dist.sample()
            # entropy += trans_dist.entropy()

            if self.value_cont:
                value_logits = self.value(inputs_step); value_dist = self.value.dist[0](value_logits[0])
                values = value_dist.sample()
                entropy += value_dist.entropy()
            else: values = self.value(inputs_step)

            returns_pred = inputs['rewards'] + values
            with tape_action:
                loss_action = util.loss_PG(action_dist, action, returns_pred, entropy, compute_dtype=self.compute_dtype)
            gradients = tape_action.gradient(loss_action, self.rep.trainable_variables + self.action.trainable_variables)
            self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.action.trainable_variables))

            rewards = rewards.write(step, inputs['rewards'][-1])
            dones = dones.write(step, inputs['dones'][-1])
            returns_updt = returns.stack()
            returns_updt = returns_updt + inputs['rewards'][-1]
            returns = returns.unstack(returns_updt)
            returns = returns.write(step, [self.float64_zero])

            loss_actions = loss_actions.write(step, loss_action)
            metric_entropy = metric_entropy.write(step, entropy)
            metric_returns_pred = metric_returns_pred.write(step, returns_pred[0])

            step += 1

        outputs = {}
        out_obs, out_actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): out_obs[i] = obs[i].stack()
        for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
        outputs['obs'], outputs['actions'], outputs['rewards'], outputs['dones'], outputs['returns'] = out_obs, out_actions, rewards.stack(), dones.stack(), returns.stack()

        loss['action'], loss['entropy'], loss['returns_pred'] = loss_actions.concat(), metric_entropy.concat(), metric_returns_pred.concat()
        return outputs, inputs, loss

    def VPN_return_learner(self, inputs, training=True):
        print("tracing -> GeneralAI VPN_return_learner")
        loss = {}
        loss_returns = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        for step in tf.range(tf.shape(inputs['dones'])[0]):
            inputs_step = {}

            obs = [None]*self.obs_spec_len
            for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
            inputs_step['obs'] = obs
            with tf.GradientTape(persistent=True) as tape_value:
                rep_logits = self.rep(inputs_step, step=step); rep_dist = self.rep.dist(rep_logits)
                inputs_step['obs'] = rep_dist.sample()

            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
            inputs_step['actions'] = action
            with tape_value:
                trans_logits = self.trans(inputs_step); trans_dist = self.trans.dist(trans_logits)
                inputs_step['obs'] = trans_dist.sample()

            returns = inputs['returns'][step:step+1]
            with tape_value:
                if self.value_cont:
                    value_logits = self.value(inputs_step); value_dist = self.value.dist[0](value_logits[0])
                    loss_return = util.loss_likelihood(value_dist, returns, compute_dtype=self.compute_dtype)
                else:
                    values = self.value(inputs_step)
                    loss_return = util.loss_diff(values, returns, compute_dtype=self.compute_dtype)
            gradients = tape_value.gradient(loss_return, self.rep.trainable_variables + self.trans.trainable_variables + self.value.trainable_variables)
            self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.value.trainable_variables))

            loss_returns = loss_returns.write(step, loss_return)

        loss['return'] = loss_returns.concat()
        return loss

    def VPN_run_episode(self, inputs, episode, training=True):
        print("tracing -> GeneralAI VPN_run_episode")
        while not inputs['dones'][-1][0]:
            self.reset_states(); outputs, inputs, loss_actor = self.VPN_actor(inputs)
            self.reset_states(); loss_return = self.VPN_return_learner(outputs)

            metrics = [episode, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                tf.math.reduce_mean(loss_actor['returns_pred']),
                tf.math.reduce_mean(loss_actor['action']),
                tf.math.reduce_mean(loss_actor['entropy']),
                tf.math.reduce_mean(loss_return['return']),
            ]
            dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

    def VPN(self):
        print("tracing -> GeneralAI VPN")
        for episode in tf.range(self.max_episodes):
            tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}
            self.VPN_run_episode(inputs, episode)





    def SPR_learner(self, inputs, num_img_steps, training=True):
        print("tracing -> GeneralAI SPR_learner")
        loss = {}
        loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_next_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_next_actions_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        steps = tf.shape(inputs['dones'])[0]
        for step in tf.range(steps):
            inputs_step = {}

            obs = [None]*self.obs_spec_len
            for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
            inputs_step['obs'] = obs
            with tf.GradientTape(persistent=True) as tape_action, tf.GradientTape(persistent=True) as tape_next_action:
                rep_logits = self.rep(inputs_step, step=step); rep_dist = self.rep.dist(rep_logits)
                inputs_step['obs'] = rep_dist.sample()

            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
            returns = inputs['returns'][step:step+1]
            with tape_action:
                action_logits = self.action(inputs_step)
                action_dist = [None]*self.action_spec_len
                for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
                loss_action = util.loss_PG(action_dist, action, returns, compute_dtype=self.compute_dtype)
            gradients = tape_action.gradient(loss_action, self.rep.trainable_variables + self.action.trainable_variables)
            self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.action.trainable_variables))
            loss_actions = loss_actions.write(step, loss_action)


            inputs_step['actions'] = action
            with tape_next_action:
                trans_logits = self.trans(inputs_step); trans_dist = self.trans.dist(trans_logits)
                inputs_step['obs'] = trans_dist.sample()

            dones = inputs['dones'][step:step+1]

            if not dones[-1][0]:
                self.reset_states(use_img=True)

                next_action = [None]*self.action_spec_len
                for i in range(self.action_spec_len): next_action[i] = inputs['actions'][i][step+1:step+2]; next_action[i].set_shape(self.action_spec[i]['step_shape'])
                with tape_next_action:
                    next_action_logits = self.action(inputs_step, use_img=True)
                    next_action_dist = [None]*self.action_spec_len
                    for i in range(self.action_spec_len): next_action_dist[i] = self.action.dist[i](next_action_logits[i])
                    loss_next_action = util.loss_likelihood(next_action_dist, next_action, compute_dtype=self.compute_dtype)
                gradients = tape_next_action.gradient(loss_next_action, self.rep.trainable_variables + self.trans.trainable_variables + self.action.trainable_variables)
                self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.action.trainable_variables))
                loss_next_actions = loss_next_actions.write(step, loss_next_action)


                loss_next_action_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
                step_img, step_img_max = step+2, step+2+num_img_steps
                while step_img < step_img_max and step_img < steps:

                    inputs_step['actions'] = next_action
                    with tf.GradientTape(persistent=True) as tape_next_action_img:
                        trans_logits = self.trans(inputs_step, use_img=True); trans_dist = self.trans.dist(trans_logits)
                        inputs_step['obs'] = trans_dist.sample()

                    next_action = [None]*self.action_spec_len
                    for i in range(self.action_spec_len): next_action[i] = inputs['actions'][i][step_img:step_img+1]; next_action[i].set_shape(self.action_spec[i]['step_shape'])
                    with tape_next_action_img:
                        next_action_logits = self.action(inputs_step, use_img=True)
                        next_action_dist = [None]*self.action_spec_len
                        for i in range(self.action_spec_len): next_action_dist[i] = self.action.dist[i](next_action_logits[i])
                        loss_next_action = util.loss_likelihood(next_action_dist, next_action, compute_dtype=self.compute_dtype)
                    gradients = tape_next_action_img.gradient(loss_next_action, self.trans.trainable_variables + self.action.trainable_variables)
                    self.net.optimizer['act'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.action.trainable_variables))
                    loss_next_action_img_accu = loss_next_action_img_accu.write(step_img-2, loss_next_action)

                    step_img += 1
                loss_next_actions_img = loss_next_actions_img.write(step, tf.math.reduce_mean(loss_next_action_img_accu.stack(), axis=0))

        loss['action'], loss['next_action'], loss['next_action_img'] = loss_actions.concat(), loss_next_actions.concat(), loss_next_actions_img.concat()
        return loss

    def SPR_run_episode(self, inputs, episode, training=True):
        print("tracing -> GeneralAI SPR_run_episode")
        while not inputs['dones'][-1][0]:
            self.reset_states(); outputs, inputs = self.PG_actor(inputs)
            self.reset_states(); loss = self.SPR_learner(outputs, num_img_steps=2)

            metrics = [episode, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                tf.math.reduce_mean(loss['action']),
                tf.math.reduce_mean(loss['next_action']),
                tf.math.reduce_mean(loss['next_action_img']),
            ]
            dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

    def SPR(self):
        print("tracing -> GeneralAI SPR")
        for episode in tf.range(self.max_episodes):
            tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}
            self.SPR_run_episode(inputs, episode)





    def MU2_img_learner(self, inputs):
        print("tracing -> GeneralAI MU2_img_learner")
        loss = {}
        loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        # metric_entropy = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        # metric_returns_pred = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        inputs_step, dones = {'obs':inputs['obs'], 'actions':self.action_zero_out}, tf.constant([[False]])
        step = tf.constant(0)
        while step < 4 and not dones[-1][0]:
        # while not dones[-1][0]:

            with tf.GradientTape(persistent=True) as tape_action:
                action_logits = self.action(inputs_step, use_img=True)
                action_dist, action = [None]*self.action_spec_len, [None]*self.action_spec_len
                for i in range(self.action_spec_len):
                    action_dist[i] = self.action.dist[i](action_logits[i])
                    action[i] = action_dist[i].sample()

            inputs_step['actions'] = action
            trans_logits = self.trans(inputs_step, use_img=True); trans_dist = self.trans.dist(trans_logits)
            inputs_step['obs'] = trans_dist.sample()
            # entropy = trans_dist.entropy()
            # entropy = tf.constant([0.0], dtype=self.compute_dtype)

            rwd_logits = self.rwd(inputs_step, use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
            done_logits = self.done(inputs_step, use_img=True); done_dist = self.done.dist[0](done_logits[0])
            rewards, dones = rwd_dist.sample(), tf.cast(done_dist.sample(), tf.bool)

            if self.value_cont:
                value_logits = self.value(inputs_step, use_img=True); value_dist = self.value.dist[0](value_logits[0])
                values = value_dist.sample()
                # entropy = entropy + value_dist.entropy()
            else: values = self.value(inputs_step, use_img=True)

            returns_pred = rewards + values
            with tape_action:
                # loss_action = util.loss_PG(action_dist, action, returns_pred, entropy, compute_dtype=self.compute_dtype)
                loss_action = util.loss_PG(action_dist, action, returns_pred, compute_dtype=self.compute_dtype)
            gradients = tape_action.gradient(loss_action, self.action.trainable_variables)
            self.net.optimizer['act'].apply_gradients(zip(gradients, self.action.trainable_variables))
            loss_actions = loss_actions.write(step, loss_action)
            # metric_entropy = metric_entropy.write(step, entropy)
            # metric_returns_pred = metric_returns_pred.write(step, returns_pred[0])

            step += 1

        # loss['action'], loss['entropy'], loss['returns_pred'] = loss_actions.stack(), metric_entropy.stack(), metric_returns_pred.stack()
        loss['action'] = loss_actions.stack()
        return loss

    def MU2_actor(self, inputs):
        print("tracing -> GeneralAI MU2_actor")
        loss = {}
        loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        # metric_entropy = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        # metric_returns_pred = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        obs, actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): obs[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['event_shape'])
        for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
        rewards = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        dones = tf.TensorArray(tf.bool, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        returns = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))

        step = tf.constant(0)
        # while step < self.max_steps and not inputs['dones'][-1][0]:
        while not inputs['dones'][-1][0]:
            inputs_step = {}
            for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])

            rep_logits = self.rep(inputs, step=step); rep_dist = self.rep.dist(rep_logits)
            inputs_step['obs'] = rep_dist.sample()

            for img_traj in tf.range(4):
                self.reset_states(use_img=True); loss_img = self.MU2_img_learner(inputs_step)
                # action_rnd = [None]*self.action_spec_len
                # for i in range(self.action_spec_len):
                #     action_rnd[i] = tf.random.uniform((self.action_spec[i]['step_shape']), minval=self.action_spec[i]['min'], maxval=self.action_spec[i]['max'], dtype=self.action_spec[i]['dtype_out'])

                loss_actions = loss_actions.write(step, tf.math.reduce_mean(loss_img['action'], axis=0))
                # metric_entropy = metric_entropy.write(step, tf.math.reduce_mean(loss_img['entropy'], axis=0))
                # metric_returns_pred = metric_returns_pred.write(step, tf.math.reduce_mean(loss_img['returns_pred'], axis=0))

            action_logits = self.action(inputs_step)
            action_dist, action, action_dis = [None]*self.action_spec_len, [None]*self.action_spec_len, [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                action_dist[i] = self.action.dist[i](action_logits[i])
                action[i] = action_dist[i].sample()
                actions[i] = actions[i].write(step, action[i][-1])
                action_dis[i] = util.discretize(action[i], self.action_spec[i], self.force_cont_action)

            np_in = tf.numpy_function(self.env_step, action_dis, self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs['obs'], inputs['rewards'], inputs['dones'] = np_in[:-2], np_in[-2], np_in[-1]

            inputs_step['actions'] = action
            trans_logits = self.trans(inputs_step); trans_dist = self.trans.dist(trans_logits)
            inputs_step['obs'] = trans_dist.sample()

            rwd_logits = self.rwd(inputs_step); rwd_dist = self.rwd.dist[0](rwd_logits[0])
            done_logits = self.done(inputs_step); done_dist = self.done.dist[0](done_logits[0])

            if self.value_cont: value_logits = self.value(inputs_step); value_dist = self.value.dist[0](value_logits[0])
            else: values = self.value(inputs_step)

            rewards = rewards.write(step, inputs['rewards'][-1])
            dones = dones.write(step, inputs['dones'][-1])
            returns_updt = returns.stack()
            returns_updt = returns_updt + inputs['rewards'][-1]
            returns = returns.unstack(returns_updt)
            returns = returns.write(step, [self.float64_zero])

            step += 1

        outputs = {}
        out_obs, out_actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): out_obs[i] = obs[i].stack()
        for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
        outputs['obs'], outputs['actions'], outputs['rewards'], outputs['dones'], outputs['returns'] = out_obs, out_actions, rewards.stack(), dones.stack(), returns.stack()

        # loss['action'], loss['entropy'], loss['returns_pred'] = loss_actions.concat(), metric_entropy.concat(), metric_returns_pred.concat()
        loss['action'] = loss_actions.concat()
        return outputs, inputs, loss

    def MU2_learner(self, inputs, num_img_steps, training=True):
        print("tracing -> GeneralAI MU2_learner")
        loss = {}
        # loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_rewards = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_dones = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_returns = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_next_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        loss_rewards_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_dones_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_returns_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_next_actions_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        steps = tf.shape(inputs['dones'])[0]
        for step in tf.range(steps):
            inputs_step = {}

            obs = [None]*self.obs_spec_len
            for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
            inputs_step['obs'] = obs
            with tf.GradientTape(persistent=True) as tape_action, tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done, tf.GradientTape(persistent=True) as tape_value, tf.GradientTape(persistent=True) as tape_next_action:
                rep_logits = self.rep(inputs_step, step=step); rep_dist = self.rep.dist(rep_logits)
                inputs_step['obs'] = rep_dist.sample()

            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
            # returns = inputs['returns'][step:step+1]
            # returns = inputs['rewards'][step:step+1] + returns
            # with tape_action:
            #     action_logits = self.action(inputs_step)
            #     action_dist = [None]*self.action_spec_len
            #     for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
            #     loss_action = util.loss_PG(action_dist, action, returns, compute_dtype=self.compute_dtype)
            # gradients = tape_action.gradient(loss_action, self.rep.trainable_variables + self.action.trainable_variables)
            # self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.action.trainable_variables))
            # loss_actions = loss_actions.write(step, loss_action)



            inputs_step['actions'] = action
            with tape_reward, tape_done, tape_value, tape_next_action:
                trans_logits = self.trans(inputs_step); trans_dist = self.trans.dist(trans_logits)
                inputs_step['obs'] = trans_dist.sample()

            rewards = inputs['rewards'][step:step+1]
            with tape_reward:
                rwd_logits = self.rwd(inputs_step); rwd_dist = self.rwd.dist[0](rwd_logits[0])
                loss_reward = util.loss_likelihood(rwd_dist, rewards, compute_dtype=self.compute_dtype)
            gradients = tape_reward.gradient(loss_reward, self.rep.trainable_variables + self.trans.trainable_variables + self.rwd.trainable_variables)
            self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.rwd.trainable_variables))
            loss_rewards = loss_rewards.write(step, loss_reward)

            dones = inputs['dones'][step:step+1]
            with tape_done:
                done_logits = self.done(inputs_step); done_dist = self.done.dist[0](done_logits[0])
                loss_done = util.loss_likelihood(done_dist, dones, compute_dtype=self.compute_dtype)
            gradients = tape_done.gradient(loss_done, self.rep.trainable_variables + self.trans.trainable_variables + self.done.trainable_variables)
            self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.done.trainable_variables))
            loss_dones = loss_dones.write(step, loss_done)

            if not dones[-1][0]:
                returns = inputs['returns'][step:step+1]
                with tape_value:
                    if self.value_cont:
                        value_logits = self.value(inputs_step); value_dist = self.value.dist[0](value_logits[0])
                        loss_return = util.loss_likelihood(value_dist, returns, compute_dtype=self.compute_dtype)
                    else:
                        values = self.value(inputs_step)
                        loss_return = util.loss_diff(values, returns, compute_dtype=self.compute_dtype)
                gradients = tape_value.gradient(loss_return, self.rep.trainable_variables + self.trans.trainable_variables + self.value.trainable_variables)
                self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.value.trainable_variables))
                loss_returns = loss_returns.write(step, loss_return)

                self.reset_states(use_img=True)

                next_action = [None]*self.action_spec_len
                for i in range(self.action_spec_len): next_action[i] = inputs['actions'][i][step+1:step+2]; next_action[i].set_shape(self.action_spec[i]['step_shape'])
                with tape_next_action:
                    next_action_logits = self.action(inputs_step, use_img=True)
                    next_action_dist = [None]*self.action_spec_len
                    for i in range(self.action_spec_len): next_action_dist[i] = self.action.dist[i](next_action_logits[i])
                    loss_next_action = util.loss_likelihood(next_action_dist, next_action, compute_dtype=self.compute_dtype)
                gradients = tape_next_action.gradient(loss_next_action, self.rep.trainable_variables + self.trans.trainable_variables + self.action.trainable_variables)
                self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.action.trainable_variables))
                loss_next_actions = loss_next_actions.write(step, loss_next_action)



                loss_reward_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
                loss_done_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
                loss_return_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
                loss_next_action_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
                step_img, step_img_max = step+1, step+1+num_img_steps
                while step_img < step_img_max and step_img < steps:

                    inputs_step['actions'] = next_action
                    with tf.GradientTape(persistent=True) as tape_reward_img, tf.GradientTape(persistent=True) as tape_done_img, tf.GradientTape(persistent=True) as tape_value_img, tf.GradientTape(persistent=True) as tape_next_action_img:
                        trans_logits = self.trans(inputs_step, use_img=True); trans_dist = self.trans.dist(trans_logits)
                        inputs_step['obs'] = trans_dist.sample()

                    rewards = inputs['rewards'][step_img:step_img+1]
                    with tape_reward_img:
                        rwd_logits = self.rwd(inputs_step, use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
                        loss_reward = util.loss_likelihood(rwd_dist, rewards, compute_dtype=self.compute_dtype)
                    gradients = tape_reward_img.gradient(loss_reward, self.trans.trainable_variables + self.rwd.trainable_variables)
                    self.net.optimizer['act'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.rwd.trainable_variables))
                    loss_reward_img_accu = loss_reward_img_accu.write(step_img-1, loss_reward)

                    dones = inputs['dones'][step_img:step_img+1]
                    with tape_done_img:
                        done_logits = self.done(inputs_step, use_img=True); done_dist = self.done.dist[0](done_logits[0])
                        loss_done = util.loss_likelihood(done_dist, dones, compute_dtype=self.compute_dtype)
                    gradients = tape_done_img.gradient(loss_done, self.trans.trainable_variables + self.done.trainable_variables)
                    self.net.optimizer['act'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.done.trainable_variables))
                    loss_done_img_accu = loss_done_img_accu.write(step_img-1, loss_done)

                    step_img += 1
                    if step_img < steps:
                        returns = inputs['returns'][step_img-1:step_img]
                        with tape_value_img:
                            if self.value_cont:
                                value_logits = self.value(inputs_step, use_img=True); value_dist = self.value.dist[0](value_logits[0])
                                loss_return = util.loss_likelihood(value_dist, returns, compute_dtype=self.compute_dtype)
                            else:
                                values = self.value(inputs_step, use_img=True)
                                loss_return = util.loss_diff(values, returns, compute_dtype=self.compute_dtype)
                        gradients = tape_value_img.gradient(loss_return, self.trans.trainable_variables + self.value.trainable_variables)
                        self.net.optimizer['act'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.value.trainable_variables))
                        loss_return_img_accu = loss_return_img_accu.write(step_img-2, loss_return)

                        next_action = [None]*self.action_spec_len
                        for i in range(self.action_spec_len): next_action[i] = inputs['actions'][i][step_img:step_img+1]; next_action[i].set_shape(self.action_spec[i]['step_shape'])
                        with tape_next_action_img:
                            next_action_logits = self.action(inputs_step, use_img=True)
                            next_action_dist = [None]*self.action_spec_len
                            for i in range(self.action_spec_len): next_action_dist[i] = self.action.dist[i](next_action_logits[i])
                            loss_next_action = util.loss_likelihood(next_action_dist, next_action, compute_dtype=self.compute_dtype)
                        gradients = tape_next_action_img.gradient(loss_next_action, self.trans.trainable_variables + self.action.trainable_variables)
                        self.net.optimizer['act'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.action.trainable_variables))
                        loss_next_action_img_accu = loss_next_action_img_accu.write(step_img-2, loss_next_action)
                loss_rewards_img = loss_rewards_img.write(step, tf.math.reduce_mean(loss_reward_img_accu.stack(), axis=0))
                loss_dones_img = loss_dones_img.write(step, tf.math.reduce_mean(loss_done_img_accu.stack(), axis=0))
                loss_returns_img = loss_returns_img.write(step, tf.math.reduce_mean(loss_return_img_accu.stack(), axis=0))
                loss_next_actions_img = loss_next_actions_img.write(step, tf.math.reduce_mean(loss_next_action_img_accu.stack(), axis=0))


        # loss['action'] = loss_actions.concat()
        loss['reward'], loss['done'], loss['return'], loss['next_action'] = loss_rewards.concat(), loss_dones.concat(), loss_returns.concat(), loss_next_actions.concat()
        loss['reward_img'], loss['done_img'], loss['return_img'], loss['next_action_img'] = loss_rewards_img.concat(), loss_dones_img.concat(), loss_returns_img.concat(), loss_next_actions_img.concat()
        return loss

    def MU2_run_episode(self, inputs, episode, training=True):
        print("tracing -> GeneralAI MU2_run_episode")
        while not inputs['dones'][-1][0]:
            self.reset_states(); outputs, inputs, loss_actor = self.MU2_actor(inputs)
            self.reset_states(); loss = self.MU2_learner(outputs, num_img_steps=4)

            metrics = [episode, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                tf.math.reduce_mean(loss_actor['action']),
                # tf.math.reduce_mean(loss['action']),
                tf.math.reduce_mean(loss['reward']), tf.math.reduce_mean(loss['reward_img']),
                tf.math.reduce_mean(loss['done']), tf.math.reduce_mean(loss['done_img']),
                tf.math.reduce_mean(loss['return']), tf.math.reduce_mean(loss['return_img']),
                tf.math.reduce_mean(loss['next_action']), tf.math.reduce_mean(loss['next_action_img']),
                # tf.math.reduce_mean(loss['reward']),
                # tf.math.reduce_mean(loss['done']),
                # tf.math.reduce_mean(loss['return']),
                # tf.math.reduce_mean(loss['next_action']),
            ]
            dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

    def MU2(self):
        print("tracing -> GeneralAI MU2")
        for episode in tf.range(self.max_episodes):
            tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}
            self.MU2_run_episode(inputs, episode)





    def MU3_img_actor(self, inputs):
        print("tracing -> GeneralAI MU3_img_actor")
        obs = tf.TensorArray(self.latent_spec['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])
        # actions = [None]*self.action_spec_len
        # for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
        rewards = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))
        returns = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))
        metric_entropy_rwd = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        metric_entropy_done = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        inputs_step = {'obs':inputs['obs'], 'actions':inputs['actions']}
        dones = tf.constant([[False]])
        # action_first, values, entropy = self.action_zero_out, tf.constant([[0.0]], dtype=self.compute_dtype), tf.constant(0.0, dtype=self.compute_dtype)

        step = tf.constant(0)
        # while step < 4 and not dones[-1][0]:
        while not dones[-1][0]:
            obs = obs.write(step, inputs_step['obs'])
            trans_logits = self.trans(inputs_step, use_img=True); trans_dist = self.trans.dist(trans_logits)
            inputs_step['obs'] = trans_dist.sample()

            rwd_logits = self.rwd(inputs_step, use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
            done_logits = self.done(inputs_step, use_img=True); done_dist = self.done.dist[0](done_logits[0])
            reward, dones = tf.cast(rwd_dist.sample(), tf.float64), tf.cast(done_dist.sample(), tf.bool)
            entropy_rwd, entropy_done = rwd_dist.entropy(), done_dist.entropy()
            metric_entropy_rwd = metric_entropy_rwd.write(step, entropy_rwd)
            metric_entropy_done = metric_entropy_done.write(step, entropy_done)

            # if self.value_cont:
            #     value_logits = self.value(inputs_step, use_img=True); value_dist = self.value.dist[0](value_logits[0])
            #     values = value_dist.sample()
            # else: values = self.value(inputs_step, use_img=True)

            rewards = rewards.write(step, reward[-1])
            returns_updt = returns.stack()
            returns_updt = returns_updt + reward[-1]
            returns = returns.unstack(returns_updt)
            returns = returns.write(step, [self.float64_zero])

            action = self.action_zero_out
            if not dones[-1][0]:
                action_logits = self.action(inputs_step, use_img=True)
                action_dist, action = [None]*self.action_spec_len, [None]*self.action_spec_len
                for i in range(self.action_spec_len):
                    # logits_rnd = tf.random.uniform(tf.shape(action_logits[i]), minval=-0.1, maxval=0.1, dtype=self.compute_dtype)
                    # logits_rnd += action_logits[i] * entropy
                    action_dist[i] = self.action.dist[i](action_logits[i])
                    action[i] = action_dist[i].sample()
                    # actions[i] = actions[i].write(step, action[i][-1])
                # if step == 0: action_first = action
            inputs_step['actions'] = action

            step += 1

        # returns = returns.write(step, [self.float64_zero])
        # returns_updt = returns.stack()
        # returns_updt = returns_updt + values[-1]
        # returns = returns.unstack(returns_updt)
        # returns_first = returns.stack()[0] + values[-1]
        # returns_first = returns.stack()[:1]

        outputs = {}
        # out_actions = [None]*self.action_spec_len
        # for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
        # outputs['actions'], outputs['returns'] = out_actions, returns.stack()
        # outputs['actions'], outputs['returns'] = action_first, returns_first
        # outputs['returns'] = returns_first
        outputs['obs'], outputs['rewards'], outputs['returns'] = obs.stack(), rewards.stack(), returns.stack()
        outputs['entropy_rwd'], outputs['entropy_done'] = tf.math.reduce_mean(metric_entropy_rwd.stack(), axis=0), tf.math.reduce_mean(metric_entropy_done.stack(), axis=0)
        return outputs

    def MU3_actor(self, inputs):
        print("tracing -> GeneralAI MU3_actor")
        loss = {}
        loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_rewards = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_dones = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        metric_entropy = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        metric_returns_pred = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        obs, actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): obs[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['event_shape'])
        for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
        rewards = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        dones = tf.TensorArray(tf.bool, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        returns = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))

        step = tf.constant(0)
        # while step < self.max_steps and not inputs['dones'][-1][0]:
        while not inputs['dones'][-1][0]:
            inputs_step = {'obs':self.latent_zero, 'actions':self.action_zero_out}
            for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])

            with tf.GradientTape(persistent=True) as tape_action, tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done:
                rep_logits = self.rep(inputs, step=step); rep_dist = self.rep.dist(rep_logits)
                # if step != 0:
                inputs_step['obs'] = rep_dist.sample()

            # action, returns_max = self.action_zero_out, self.float_min
            # for img_traj in tf.range(4):
            #     self.reset_states(use_img=True); outputs_img = self.MU3_img_actor(inputs_step)
            #     if outputs_img['returns'] > returns_max:
            #         returns_max = outputs_img['returns']
            #         action = outputs_img['actions']
            # with tape_action:
            #     action_logits = self.action(inputs_step)
            #     action_dist = [None]*self.action_spec_len
            #     for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])


            # TODO can I add ARS/MCTS here?
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                action[i] = tf.random.uniform((self.action_spec[i]['step_shape']), minval=self.action_spec[i]['min'], maxval=self.action_spec[i]['max'], dtype=self.action_spec[i]['dtype_out'])


            # # with tape_action:
            # action_logits = self.action(inputs_step)
            # action_dist, action = [None]*self.action_spec_len, [None]*self.action_spec_len
            # for i in range(self.action_spec_len):
            #     action_dist[i] = self.action.dist[i](action_logits[i])
            #     action[i] = action_dist[i].sample()
            # self.action.reset_states(use_img=True)
            # inputs_step['actions'] = action
            # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
            # outputs_img = self.MU3_img_actor(inputs_step)

            # action_logits = self.action(inputs_step, store_memory=False, use_img=True)
            # # action_dist, action = [None]*self.action_spec_len, [None]*self.action_spec_len
            # for i in range(self.action_spec_len):
            #     action_dist[i] = self.action.dist[i](action_logits[i])
            #     # action[i] = action_dist[i].sample(); action[i].set_shape(self.action_spec[i]['step_shape'])
            #     action[i] = action_dist[i].sample()
            # inputs_step['actions'] = action
            # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
            # outputs_img = self.MU3_img_actor(inputs_step)

            # action_logits = self.action(inputs_step, store_memory=False, use_img=True)
            # # action_dist, action = [None]*self.action_spec_len, [None]*self.action_spec_len
            # for i in range(self.action_spec_len):
            #     action_dist[i] = self.action.dist[i](action_logits[i])
            #     # action[i] = action_dist[i].sample(); action[i].set_shape(self.action_spec[i]['step_shape'])
            #     action[i] = action_dist[i].sample()
            # inputs_step['actions'] = action
            # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
            # outputs_img = self.MU3_img_actor(inputs_step)

            # with tape_action:
            #     action_logits = self.action(inputs_step, store_memory=False, use_img=True)
            #     # action_dist, action = [None]*self.action_spec_len, [None]*self.action_spec_len
            #     for i in range(self.action_spec_len):
            #         action_dist[i] = self.action.dist[i](action_logits[i])
            #         # action[i] = action_dist[i].sample(); action[i].set_shape(self.action_spec[i]['step_shape'])
            #         action[i] = action_dist[i].sample()
            # inputs_step['actions'] = action
            # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
            # outputs_img = self.MU3_img_actor(inputs_step)


            action_dis = [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                actions[i] = actions[i].write(step, action[i][-1])
                action_dis[i] = util.discretize(action[i], self.action_spec[i], self.force_cont_action)

            np_in = tf.numpy_function(self.env_step, action_dis, self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs['obs'], inputs['rewards'], inputs['dones'] = np_in[:-2], np_in[-2], np_in[-1]

            # entropy = tf.constant([0.0], dtype=self.compute_dtype)
            # inputs_step['actions'] = action
            with tape_reward, tape_done:
                trans_logits = self.trans(inputs_step); trans_dist = self.trans.dist(trans_logits)
                inputs_step['obs'] = trans_dist.sample()
            # entropy += trans_dist.entropy()

            # if self.value_cont:
            #     value_logits = self.value(inputs_step); value_dist = self.value.dist[0](value_logits[0])
            #     values = value_dist.sample()
            #     entropy += value_dist.entropy()
            # else: values = self.value(inputs_step)

            # if action[0] == tf.cast(inputs['obs'][0], dtype=tf.int32):
            #     tf.print('test')

            # values, entropy = outputs_img['returns'][:1], outputs_img['entropy_rwd']
            # returns_pred = inputs['rewards'] + values
            # # returns_pred = inputs['rewards']
            # with tape_action:
            #     # loss_action = util.loss_PG(action_dist, action, returns_pred, entropy, compute_dtype=self.compute_dtype)
            #     loss_action = util.loss_PG(action_dist, action, returns_pred, compute_dtype=self.compute_dtype)
            #     # loss_action = util.loss_likelihood(action_dist, action, compute_dtype=self.compute_dtype)
            # gradients = tape_action.gradient(loss_action, self.rep.trainable_variables + self.action.trainable_variables)
            # self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.action.trainable_variables))
            # loss_actions = loss_actions.write(step, loss_action)
            # metric_entropy = metric_entropy.write(step, entropy)
            # metric_returns_pred = metric_returns_pred.write(step, returns_pred[0])


            with tape_reward:
                rwd_logits = self.rwd(inputs_step); rwd_dist = self.rwd.dist[0](rwd_logits[0])
                loss_reward = util.loss_likelihood(rwd_dist, inputs['rewards'], compute_dtype=self.compute_dtype)
            gradients = tape_reward.gradient(loss_reward, self.rep.trainable_variables + self.trans.trainable_variables + self.rwd.trainable_variables)
            self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.rwd.trainable_variables))
            loss_rewards = loss_rewards.write(step, loss_reward)

            with tape_done:
                done_logits = self.done(inputs_step); done_dist = self.done.dist[0](done_logits[0])
                loss_done = util.loss_likelihood(done_dist, inputs['dones'], compute_dtype=self.compute_dtype)
            gradients = tape_done.gradient(loss_done, self.rep.trainable_variables + self.trans.trainable_variables + self.done.trainable_variables)
            self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.done.trainable_variables))
            loss_dones = loss_dones.write(step, loss_done)


            rewards = rewards.write(step, inputs['rewards'][-1])
            dones = dones.write(step, inputs['dones'][-1])
            returns_updt = returns.stack()
            returns_updt = returns_updt + inputs['rewards'][-1]
            returns = returns.unstack(returns_updt)
            returns = returns.write(step, [self.float64_zero])

            step += 1

        outputs = {}
        out_obs, out_actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): out_obs[i] = obs[i].stack()
        for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
        outputs['obs'], outputs['actions'], outputs['rewards'], outputs['dones'], outputs['returns'] = out_obs, out_actions, rewards.stack(), dones.stack(), returns.stack()

        loss['action'], loss['reward'], loss['done'], loss['entropy'], loss['returns_pred'] = loss_actions.concat(), loss_rewards.concat(), loss_dones.concat(), metric_entropy.concat(), metric_returns_pred.concat()
        return outputs, inputs, loss

    def MU3_dyn_learner(self, inputs, training=True):
        print("tracing -> GeneralAI MU3_dyn_learner")
        loss = {}
        loss_rewards = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_dones = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        obs = [None]*self.obs_spec_len
        for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][:1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
        inputs_step = {'obs':obs, 'actions':self.action_zero_out}
        rep_logits = self.rep(inputs_step, step=0); rep_dist = self.rep.dist(rep_logits)
        inputs_step['obs'] = rep_dist.sample()

        for step in tf.range(tf.shape(inputs['dones'])[0]):
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
            inputs_step['actions'] = action
            with tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done:
                trans_logits = self.trans(inputs_step); trans_dist = self.trans.dist(trans_logits)
                inputs_step['obs'] = trans_dist.sample()

            with tape_reward:
                rwd_logits = self.rwd(inputs_step); rwd_dist = self.rwd.dist[0](rwd_logits[0])
                loss_reward = util.loss_likelihood(rwd_dist, inputs['rewards'][step:step+1], compute_dtype=self.compute_dtype)
            gradients = tape_reward.gradient(loss_reward, self.trans.trainable_variables)
            self.net.optimizer['act'].apply_gradients(zip(gradients, self.trans.trainable_variables))
            loss_rewards = loss_rewards.write(step, loss_reward)

            with tape_done:
                done_logits = self.done(inputs_step); done_dist = self.done.dist[0](done_logits[0])
                loss_done = util.loss_likelihood(done_dist, inputs['dones'][step:step+1], compute_dtype=self.compute_dtype)
            gradients = tape_done.gradient(loss_done, self.trans.trainable_variables)
            self.net.optimizer['act'].apply_gradients(zip(gradients, self.trans.trainable_variables))
            loss_dones = loss_dones.write(step, loss_done)

        loss['reward'], loss['done'] = loss_rewards.concat(), loss_dones.concat()
        return loss

    def MU3_run_episode(self, inputs, episode, training=True):
        print("tracing -> GeneralAI MU3_run_episode")
        while not inputs['dones'][-1][0]:
            self.reset_states(); outputs, inputs, loss_actor = self.MU3_actor(inputs)
            self.reset_states(); loss_dyn = self.MU3_dyn_learner(outputs)

            metrics = [episode, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                tf.math.reduce_mean(loss_actor['returns_pred']),
                tf.math.reduce_mean(loss_actor['action']),
                tf.math.reduce_mean(loss_actor['reward']), tf.math.reduce_mean(loss_actor['done']),
                tf.math.reduce_mean(loss_dyn['reward']), tf.math.reduce_mean(loss_dyn['done']),
                tf.math.reduce_mean(loss_actor['entropy']),
            ]
            # TODO add simulation episode time
            if self.trader: metrics += [tf.math.reduce_mean(tf.concat([outputs['obs'][3],inputs['obs'][3]],0)), inputs['obs'][3][-1][0],
                tf.math.reduce_mean(tf.concat([outputs['obs'][4],inputs['obs'][4]],0)), tf.math.reduce_mean(tf.concat([outputs['obs'][5],inputs['obs'][5]],0)),
                inputs['obs'][0][-1][0] - outputs['obs'][0][0][0],]
            dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

    def MU3(self):
        print("tracing -> GeneralAI MU3")
        for episode in tf.range(self.max_episodes):
            tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}
            self.MU3_run_episode(inputs, episode)




def params(): pass
load_model, save_model = False, False
max_episodes = 10
learn_rate = 2e-6 # 5 = testing, 6 = more stable/slower # tf.experimental.numpy.finfo(tf.float64).eps
entropy_contrib = 0 # 1e-8
returns_disc = 1.0
value_cont = True
force_cont_obs, force_cont_action = False, False
latent_size = 128
latent_dist = 0 # 0 = deterministic, 1 = categorical, 2 = continuous
net_attn_io = True
aio_max_latents = 32
attn_mem_multi = 4 # attn_img_base # max_steps must be power of this!
aug_data_step, aug_data_pos = True, False

device_type = 'GPU' # use GPU for large networks (over 8 total net blocks?) or output data (512 bytes?)
device_type = 'CPU'

machine, device, extra = 'dev', 1, '_rp200_gen012_pgl-2e9_a-6e4-std2' # _repL1 _gen0123 _dyn1279 _rp200-rnd _img _prs2 _wd7 _train _RfB _entropy3 _mae _perO-NR-NT-G-Nrez _rez-rezoR-rezoT-rezoG _mixlog-abs-log1p-Nreparam _obs-tsBoxF-dataBoxI_round _Nexp-Ne9-Nefmp36-Nefmer154-Nefme308-emr-Ndiv _MUimg-entropy-values-policy-Netoe _AC-Nonestep-aing _stepE _cncat

trader, env_async, env_async_clock, env_async_speed = False, False, 0.001, 160.0
env_name, max_steps, env_render, env = 'CartPole', 256, False, gym.make('CartPole-v0') # ; env.observation_space.dtype = np.dtype('float64') # (4) float32    ()2 int64    200  195.0
# env_name, max_steps, env_render, env = 'CartPole', 512, False, gym.make('CartPole-v1') # ; env.observation_space.dtype = np.dtype('float64') # (4) float32    ()2 int64    500  475.0
# env_name, max_steps, env_render, env = 'LunarLand', 1024, False, gym.make('LunarLander-v2') # (8) float32    ()4 int64    1000  200
# env_name, max_steps, env_render, env = 'Copy', 256, False, gym.make('Copy-v0') # DuplicatedInput-v0 RepeatCopy-v0 Reverse-v0 ReversedAddition-v0 ReversedAddition3-v0 # ()6 int64    [()2,()2,()5] int64    200  25.0
# env_name, max_steps, env_render, env = 'ProcgenChaser', 1024, False, gym.make('procgen-chaser-v0') # (64,64,3) uint8    ()15 int64    1000 None
# env_name, max_steps, env_render, env = 'ProcgenCaveflyer', 1024, False, gym.make('procgen-caveflyer-v0') # (64,64,3) uint8    ()15 int64    1000 None
# env_name, max_steps, env_render, env = 'Tetris', 22528, False, gym.make('ALE/Tetris-v5') # (210,160,3) uint8    ()18 int64    21600 None
# env_name, max_steps, env_render, env = 'MontezumaRevenge', 22528, False, gym.make('MontezumaRevengeNoFrameskip-v4') # (210,160,3) uint8    ()18 int64    400000 None
# env_name, max_steps, env_render, env = 'MsPacman', 22528, False, gym.make('MsPacmanNoFrameskip-v4') # (210,160,3) uint8    ()9 int64    400000 None

# env_name, max_steps, env_render, env = 'CartPoleCont', 256, False, gym.make('CartPoleContinuousBulletEnv-v0'); env.observation_space.dtype = np.dtype('float64') # (4) float32    (1) float32    200  190.0
# env_name, max_steps, env_render, env = 'LunarLandCont', 1024, False, gym.make('LunarLanderContinuous-v2') # (8) float32    (2) float32    1000  200
# import envs_local.bipedal_walker as env_; env_name, max_steps, env_render, env = 'BipedalWalker', 2048, False, env_.BipedalWalker()
# env_name, max_steps, env_render, env = 'Hopper', 1024, False, gym.make('HopperBulletEnv-v0') # (15) float32    (3) float32    1000  2500.0
# env_name, max_steps, env_render, env = 'RacecarZed', 1024, False, gym.make('RacecarZedBulletEnv-v0') # (10,100,4) uint8    (2) float32    1000  5.0

# from pettingzoo.butterfly import pistonball_v4; env_name, max_steps, env_render, env = 'PistonBall', 1, False, pistonball_v4.env()

# import envs_local.random_env as env_; env_name, max_steps, env_render, env = 'TestRnd', 16, False, env_.RandomEnv(True)
# import envs_local.data_env as env_; env_name, max_steps, env_render, env = 'DataShkspr', 64, False, env_.DataEnv('shkspr')
# # import envs_local.data_env as env_; env_name, max_steps, env_render, env = 'DataMnist', 64, False, env_.DataEnv('mnist')
# import gym_trader; tenv = 2; env_name, max_steps, env_render, env, trader = 'Trader'+str(tenv), 1024*4, False, gym.make('Trader-v0', agent_id=device, env=tenv), True

# max_steps = 4 # max replay buffer or train interval or bootstrap

# arch = 'TEST' # testing architechures
# arch = 'PG' # Policy Gradient agent, PG loss
# arch = 'AC' # Actor Critic, PG and advantage loss
# arch = 'TRANS' # learned Transition dynamics, autoregressive likelihood loss
# arch = 'MU' # Dreamer/planner w/imagination (DeepMind MuZero)
# arch = 'VPN' # Value Prediction Network
# arch = 'SPR' # Self Predictive Representations
# arch = 'MU2' # Dreamer/planner w/imagination
# arch = 'MU3' # Dreamer/planner w/imagination
# arch = 'DREAM' # full World Model w/imagination (DeepMind Dreamer)

if __name__ == '__main__':
    # TODO add keyboard control so can stop

    ## manage multiprocessing
    # # setup ctrl,data,param sharing
    # # start agents (real+dreamers)
    # agent = Agent(model)
    # # agent_process = mp.Process(target=agent.vivify, name='AGENT', args=(lock_print, process_ctrl, weights_shared))
    # # agent_process.start()
    # # quit on keyboard (space = save, esc = no save)
    # process_ctrl.value = 0
    # agent_process.join()

    if env_async: import envs_local.async_wrapper as envaw_; env_name, env = env_name+'-asyn', envaw_.AsyncWrapperEnv(env, env_async_clock, env_async_speed, env_render)
    with tf.device("/device:{}:{}".format(device_type,(device if device_type=='GPU' else 0))):
        model = GeneralAI(arch, env, trader, env_render, max_episodes, max_steps, learn_rate, entropy_contrib, returns_disc, value_cont, force_cont_obs, force_cont_action, latent_size, latent_dist, net_attn_io, aio_max_latents, attn_mem_multi, aug_data_step, aug_data_pos)
        name = "gym-{}-{}-{}".format(arch, env_name, ['Ldet','Lcat','Lcon'][latent_dist])

        ## debugging
        # model.build(()); model.action.summary(); quit(0)
        # inputs = {'obs':model.obs_zero, 'rewards':tf.constant([[0]],tf.float64), 'dones':tf.constant([[False]],tf.bool)}
        # # inp_sig = [[[tf.TensorSpec(shape=None, dtype=tf.float32)], tf.TensorSpec(shape=None, dtype=tf.float64), tf.TensorSpec(shape=None, dtype=tf.bool)]]
        # # model.AC_actor = tf.function(model.AC_actor, input_signature=inp_sig, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
        # model.AC_actor = tf.function(model.AC_actor, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        # self.AC_actor = tf.function(self.AC_actor)
        # print(tf.autograph.to_code(model.MU_run_episode, recursive=True, experimental_optional_features=tf.autograph.experimental.Feature.LISTS)); quit(0)
        # # print(tf.autograph.to_code(model.AC_actor.python_function, experimental_optional_features=tf.autograph.experimental.Feature.LISTS)); quit(0)
        # print(model.AC_actor.get_concrete_function(inputs)); quit(0)
        # print(model.AC_actor.get_concrete_function(inputs).graph.as_graph_def()); quit(0)
        # obs, reward, done = env.reset(), 0.0, False
        # # test = model.AC_actor.python_function(inputs)
        # test = model.AC_actor(inputs)
        # print(test); quit(0)


        ## load models
        model_files, name_arch = {}, ""
        for net in model.layers:
            model_name = "{}-{}-a{}".format(net.net_arch, machine, device)
            model_file = "{}/tf-data-models-local/{}.h5".format(curdir, model_name); loaded_model = False
            model_files[net.name] = model_file
            if (load_model or net.name == 'M') and tf.io.gfile.exists(model_file):
                net.load_weights(model_file, by_name=True, skip_mismatch=True)
                print("LOADED {} weights from {}".format(net.name, model_file)); loaded_model = True
            name_opt = "-O{}{}".format(net.opt_spec['type'], ('' if net.opt_spec['schedule_type']=='' else '-S'+net.opt_spec['schedule_type'])) if hasattr(net, 'opt_spec') else ''
            name_arch += "   {}{}-{}".format(net.net_arch, name_opt, 'load' if loaded_model else 'new')


        ## run
        arch_run = getattr(model, arch)
        t1_start = time.perf_counter_ns()
        arch_run()
        total_time = (time.perf_counter_ns() - t1_start) / 1e9 # seconds
        env.close()


        ## metrics
        metrics_loss = model.metrics_loss
        for loss_group in metrics_loss.values():
            for k in loss_group.keys():
                for j in range(len(loss_group[k])): loss_group[k][j] = np.mean(loss_group[k][j])
        # TODO np.mean, reduce size if above 200,000 episodes

        name = "{}-{}-a{}{}-{}".format(name, machine, device, extra, time.strftime("%y-%m-%d-%H-%M-%S"))
        total_steps = int(np.sum(metrics_loss['1steps']['steps+']))
        step_time = total_time/total_steps
        title = "{}    [{}-{}] {}\ntime:{}    steps:{}    t/s:{:.8f}".format(name, device_type, tf.keras.backend.floatx(), name_arch, util.print_time(total_time), total_steps, step_time)
        title += "     |     lr:{}    dis:{}    en:{}    al:{}    am:{}    ms:{}".format(learn_rate, returns_disc, entropy_contrib, aio_max_latents, attn_mem_multi, max_steps)
        title += "     |     a-clk:{}    a-spd:{}    aug:{}{}    aio:{}".format(env_async_clock, env_async_speed, ('S' if aug_data_step else ''), ('P' if aug_data_pos else ''), ('Y' if net_attn_io else 'N')); print(title)

        import matplotlib as mpl
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue','lightblue','green','lime','red','lavender','turquoise','cyan','magenta','salmon','yellow','gold','black','brown','purple','pink','orange','teal','coral','darkgreen','tan'])
        plt.figure(num=name, figsize=(34, 18), tight_layout=True)
        xrng, i, vplts, lim = np.arange(0, max_episodes, 1), 0, 0, 0.01
        for loss_group_name in metrics_loss.keys(): vplts += int(loss_group_name[0])

        for loss_group_name, loss_group in metrics_loss.items():
            rows, col, m_min, m_max, combine = int(loss_group_name[0]), 0, [0]*len(loss_group), [0]*len(loss_group), loss_group_name.endswith('*')
            if combine: spg = plt.subplot2grid((vplts, 1), (i, 0), rowspan=rows, xlim=(0, max_episodes)); plt.grid(axis='y',alpha=0.3)
            for metric_name, metric in loss_group.items():
                metric = np.asarray(metric, np.float64); m_min[col], m_max[col] = np.nanquantile(metric, lim), np.nanquantile(metric, 1.0-lim)
                if not combine: spg = plt.subplot2grid((vplts, len(loss_group)), (i, col), rowspan=rows, xlim=(0, max_episodes), ylim=(m_min[col], m_max[col])); plt.grid(axis='y',alpha=0.3)
                # plt.plot(xrng, talib.EMA(metric, timeperiod=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
                # plt.plot(xrng, bottleneck.move_mean(metric, window=max_episodes//10+2, min_count=1), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
                if metric_name.startswith('-'): plt.plot(xrng, metric, alpha=1.0, label=metric_name)
                else: plt.plot(xrng, util.ewma(metric, window=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
                plt.ylabel('value'); plt.legend(loc='upper left'); col+=1
            if combine: spg.set_ylim(np.min(m_min), np.max(m_max))
            if i == 0: plt.title(title)
            i+=rows
        plt.show()


        ## save models
        if save_model:
            for net in model.layers:
                model_file = model_files[net.name]
                net.save_weights(model_file)
                print("SAVED {} weights to {}".format(net.name, model_file))
