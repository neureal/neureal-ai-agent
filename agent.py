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
    def __init__(self, arch, env, trader, env_render, max_episodes, max_steps, learn_rate, value_cont, latent_size, latent_dist, mixture_multi, net_attn_io, aio_max_latents, attn_mem_base, aug_data_step, aug_data_pos):
        super(GeneralAI, self).__init__()
        compute_dtype = tf.dtypes.as_dtype(self.compute_dtype)
        self.float_min = tf.constant(compute_dtype.min, compute_dtype)
        self.float_maxroot = tf.constant(tf.math.sqrt(compute_dtype.max), compute_dtype)
        self.float_eps = tf.constant(tf.experimental.numpy.finfo(compute_dtype).eps, compute_dtype)
        self.float64_eps = tf.constant(tf.experimental.numpy.finfo(tf.float64).eps, tf.float64)
        self.float_eps_max = tf.constant(1.0 / self.float_eps, compute_dtype)
        self.float_log_min = tf.constant(tf.math.log(self.float_eps), compute_dtype)
        self.loss_scale = tf.math.exp(tf.math.log(self.float_eps_max) * (1/2))
        self.compute_zero, self.int32_max, self.int32_maxbit, self.int32_zero, self.float64_zero = tf.constant(0, compute_dtype), tf.constant(tf.int32.max, tf.int32), tf.constant(1073741824, tf.int32), tf.constant(0, tf.int32), tf.constant(0, tf.float64)

        self.arch, self.env, self.trader, self.env_render, self.value_cont = arch, env, trader, env_render, value_cont
        self.max_episodes, self.max_steps, self.learn_rate, self.attn_mem_base = tf.constant(max_episodes, tf.int32), tf.constant(max_steps, tf.int32), tf.constant(learn_rate, tf.float64), tf.constant(attn_mem_base, tf.int32)
        self.dist_prior = tfp.distributions.Independent(tfp.distributions.Logistic(loc=tf.zeros(latent_size, dtype=self.compute_dtype), scale=10.0), reinterpreted_batch_ndims=1)
        # self.dist_prior = tfp.distributions.Independent(tfp.distributions.Uniform(low=tf.cast(tf.fill(latent_size,-10), dtype=self.compute_dtype), high=10), reinterpreted_batch_ndims=1)
        self.initializer = tf.keras.initializers.GlorotUniform()

        self.obs_spec, self.obs_zero, _ = gym_util.get_spec(env.observation_space, space_name='obs', compute_dtype=self.compute_dtype, net_attn_io=net_attn_io, aio_max_latents=aio_max_latents, mixture_multi=mixture_multi)
        self.action_spec, _, self.action_zero_out = gym_util.get_spec(env.action_space, space_name='actions', compute_dtype=self.compute_dtype, mixture_multi=mixture_multi)
        self.obs_spec_len, self.action_spec_len = len(self.obs_spec), len(self.action_spec)
        self.gym_step_shapes = [feat['step_shape'] for feat in self.obs_spec] + [tf.TensorShape((1,1)), tf.TensorShape((1,1))]
        self.gym_step_dtypes = [feat['dtype'] for feat in self.obs_spec] + [tf.float64, tf.bool]
        self.rewards_zero, self.dones_zero = tf.constant([[0]],tf.float64), tf.constant([[False]],tf.bool)
        self.step_zero, self.step_size_one = tf.constant([[0]]), tf.constant([[1]])

        self.attn_img_scales = int(np.log(max_steps) / np.log(attn_mem_base)) # TODO remove
        self.attn_img_step_sizes = [None]*(self.attn_img_scales)
        for step_scale in range(self.attn_img_scales): self.attn_img_step_sizes[step_scale] = tf.math.pow(self.attn_mem_base, step_scale+1)
        # self.attn_img_step_sizes[-1] = tf.math.pow(self.attn_mem_base, int(np.log2(max_steps))) # TODO remove
        self.attn_img_step_sizesT = tf.concat(self.attn_img_step_sizes, axis=0)
        if self.attn_img_step_sizesT.shape == (): self.attn_img_step_sizesT = tf.reshape(self.attn_img_step_sizesT, (1,))
        self.attn_img_step_locs = max_steps - tf.cast(max_steps / self.attn_img_step_sizesT, tf.int32)
        self.step_size_max = tf.reshape(self.attn_img_step_sizesT[-1],(1,1))

        net_attn, net_lstm = True, False

        latent_spec = {'dtype':compute_dtype, 'latent_size':latent_size, 'num_latents':1, 'max_latents':aio_max_latents}
        # latent_spec.update({'inp':latent_size*4, 'midp':latent_size*2, 'outp':latent_size*4, 'evo':int(latent_size/2)})
        latent_spec.update({'inp':512, 'midp':256, 'outp':512, 'evo':int(latent_size/2)})
        if latent_dist == 'd': latent_spec.update({'dist_type':'d', 'num_components':latent_size, 'event_shape':(latent_size,)}) # deterministic
        if latent_dist == 'c': latent_spec.update({'dist_type':'c', 'num_components':0, 'event_shape':(latent_size, latent_size)}) # categorical
        if latent_dist == 'mx': latent_spec.update({'dist_type':'mx', 'num_components':int(latent_size/16), 'event_shape':(latent_size,)}) # continuous

        # self.obs_spec += [{'space_name':'rewards', 'name':'', 'event_shape':(1,), 'event_size':1, 'channels':1, 'num_latents':1}]
        if aug_data_step: self.obs_spec += [{'space_name':'step', 'name':'', 'event_shape':(1,), 'event_size':1, 'channels':1, 'num_latents':1}]
        inputs = {'obs':self.obs_zero, 'rewards':[self.rewards_zero], 'step':[self.step_zero], 'step_size':[self.step_size_one]}

        if arch in ('MU4',):
            learn_rate_rep = tf.constant(2e-8, tf.float64) # self.learn_rate
            opt_spec = [
                {'name':'act', 'type':'a', 'schedule_type':'', 'learn_rate':learn_rate_rep, 'float_eps':self.float_eps},
                {'name':'PG', 'type':'a', 'schedule_type':'', 'learn_rate':learn_rate_rep, 'float_eps':self.float_eps},
                {'name':'PGL', 'type':'a', 'schedule_type':'', 'learn_rate':learn_rate_rep, 'float_eps':self.float_eps},
                {'name':'trans', 'type':'a', 'schedule_type':'', 'learn_rate':learn_rate_rep, 'float_eps':self.float_eps},
                {'name':'rwd', 'type':'a', 'schedule_type':'', 'learn_rate':learn_rate_rep, 'float_eps':self.float_eps},
                {'name':'done', 'type':'a', 'schedule_type':'', 'learn_rate':learn_rate_rep, 'float_eps':self.float_eps},
            ]
            self.rep = nets.ArchTrans('RN', inputs, opt_spec, [], self.obs_spec, latent_spec, obs_latent=False, net_blocks=0, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=None, aug_data_pos=aug_data_pos); outputs = self.rep(inputs)
            self.rep.optimizer_weights = []
            for spec in opt_spec: self.rep.optimizer_weights += util.optimizer_build(self.rep.optimizer[spec['name']], self.rep.trainable_variables)
            util.net_build(self.rep, self.initializer)
            rep_dist = self.rep.dist(outputs); self.latent_zero = tf.zeros_like(rep_dist.sample(), dtype=latent_spec['dtype'])
            latent_spec.update({'step_shape':self.latent_zero.shape}); self.latent_spec = latent_spec

            opt_spec = [{'name':'action', 'type':'a', 'schedule_type':'', 'learn_rate':tf.constant(6e-6, tf.float64), 'float_eps':self.float_eps}]; stats_spec = [{'name':'rwd', 'b1':0.99, 'b2':0.99, 'dtype':tf.float64}, {'name':'loss', 'b1':0.9, 'b2':0.9, 'dtype':compute_dtype}]
            self.action = nets.ArchGen('AN', self.latent_zero, opt_spec, stats_spec, self.action_spec, latent_spec, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=max_steps); outputs = self.action(self.latent_zero)
            self.action.optimizer_weights = util.optimizer_build(self.action.optimizer['action'], self.action.trainable_variables)
            util.net_build(self.action, self.initializer)
            # thresh = [2e-5,2e-3]; thresh_rates = [77,57,44] # 2e-12 107, 2e-10 89, 2e-8 71, 2e-7 62, 2e-6 53, 2e-5 44, 2e-4 35, 2e-3 26, 2e-2 17
            # self.action_get_learn_rate = util.LearnRateThresh(thresh, thresh_rates)

        if arch in ('MU4',):
            opt_spec = [{'name':'actionL', 'type':'a', 'schedule_type':'', 'learn_rate':tf.constant(2e-9, tf.float64), 'float_eps':self.float_eps}]; stats_spec = [{'name':'rwd', 'b1':0.99, 'b2':0.99, 'dtype':tf.float64}, {'name':'loss', 'b1':0.9, 'b2':0.9, 'dtype':compute_dtype}]
            self.actionL = nets.ArchGen('ANL', self.latent_zero, opt_spec, stats_spec, self.action_spec, latent_spec, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=max_steps); outputs = self.actionL(self.latent_zero)
            self.actionL.optimizer_weights = util.optimizer_build(self.actionL.optimizer['actionL'], self.actionL.trainable_variables)
            util.net_build(self.actionL, self.initializer)

        if arch in ('MU4',):
            inputs['obs'], inputs['actions'] = self.latent_zero, self.action_zero_out
            trans_spec = self.action_spec + [{'space_name':'step_size', 'name':'', 'event_shape':(1,), 'event_size':1, 'channels':1, 'num_latents':1}]
            latent_spec_trans = latent_spec.copy(); latent_spec_trans.update({'dist_type':'mx', 'num_components':int(latent_size/16), 'event_shape':(latent_size,)}) # continuous
            opt_spec = [{'name':'trans', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps}]
            self.trans = nets.ArchTrans('TN', inputs, opt_spec, [], trans_spec, latent_spec_trans, obs_latent=True, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=max_steps, aug_data_pos=aug_data_pos); outputs = self.trans(inputs)
            self.trans.optimizer_weights = util.optimizer_build(self.trans.optimizer['trans'], self.trans.trainable_variables)
            util.net_build(self.trans, self.initializer)

        if arch in ('MU4',):
            inputs['obs'], inputs['return_goal'] = self.latent_zero, [self.rewards_zero]
            act_spec = [{'space_name':'return_goal', 'name':'', 'event_shape':(1,), 'event_size':1, 'channels':1, 'num_latents':1}]
            opt_spec = [{'name':'act', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps}]; stats_spec = [{'name':'rwd', 'b1':0.99, 'b2':0.99, 'dtype':tf.float64}, {'name':'loss', 'b1':0.99, 'b2':0.99, 'dtype':compute_dtype}]
            self.act = nets.ArchFull('ACT', inputs, opt_spec, stats_spec, act_spec, self.action_spec, latent_spec, obs_latent=True, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=max_steps, aug_data_pos=aug_data_pos); outputs = self.act(inputs)
            self.act.optimizer_weights = util.optimizer_build(self.act.optimizer['act'], self.act.trainable_variables)
            util.net_build(self.act, self.initializer)
            thresh = [15,75,150]; thresh_rates = [80,71,62,53] # 2e-12 107, 2e-10 89, 2e-8 71, 2e-7 62, 2e-6 53, 2e-5 44, 2e-4 35, 2e-3 26, 2e-2 17
            self.act_get_learn_rate = util.LearnRateThresh(thresh, thresh_rates)

        if arch in ('MU4',):
            reward_spec = [{'space_name':'reward', 'name':'', 'dtype':tf.float64, 'dtype_out':compute_dtype, 'min':0, 'max':1, 'dist_type':'mx', 'num_components':16, 'event_shape':(1,), 'step_shape':tf.TensorShape((1,1))}]
            opt_spec = [{'name':'rwd', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps}]
            self.rwd = nets.ArchGen('RW', self.latent_zero, opt_spec, [], reward_spec, latent_spec, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=None); outputs = self.rwd(self.latent_zero)
            self.rwd.optimizer_weights = util.optimizer_build(self.rwd.optimizer['rwd'], self.rwd.trainable_variables)
            util.net_build(self.rwd, self.initializer)

            done_spec = [{'space_name':'done', 'name':'', 'dtype':tf.bool, 'dtype_out':tf.int32, 'min':0, 'max':1, 'dist_type':'c', 'num_components':2, 'event_shape':(1,), 'step_shape':tf.TensorShape((1,1))}]
            opt_spec = [{'name':'done', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps}]
            self.done = nets.ArchGen('DO', self.latent_zero, opt_spec, [], done_spec, latent_spec, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=None); outputs = self.done(self.latent_zero)
            self.done.optimizer_weights = util.optimizer_build(self.done.optimizer['done'], self.done.trainable_variables)
            util.net_build(self.done, self.initializer)

        # opt_spec = [{'name':'meta', 'type':'a', 'schedule_type':'', 'learn_rate':tf.constant(2e-5, tf.float64), 'float_eps':self.float_eps}]; stats_spec = [{'name':'loss', 'b1':0.99, 'b2':0.99, 'dtype':compute_dtype}]
        # inputs_meta = {'obs':[tf.constant([[0,0,0]],compute_dtype)]}; meta_spec_in = [{'space_name':'obs', 'name':'', 'event_shape':(3,), 'event_size':1, 'channels':3, 'num_latents':1}]
        # self.meta_spec = [{'space_name':'meta', 'name':'', 'dtype':tf.float64, 'dtype_out':compute_dtype, 'min':self.float_eps, 'max':self.learn_rate, 'dist_type':'mx', 'num_components':8, 'event_shape':(1,), 'step_shape':tf.TensorShape((1,1))}]
        # self.meta = nets.ArchFull('M', inputs_meta, opt_spec, stats_spec, meta_spec_in, self.meta_spec, latent_spec, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io); outputs = self.meta(inputs_meta)
        # self.meta.optimizer_weights = util.optimizer_build(self.meta.optimizer['meta'], self.meta.trainable_variables)
        # util.net_build(self.meta, self.initializer)


        self.metrics_spec()
        # TF bug that wont set graph options with tf.function decorator inside a class
        self.reset_states = tf.function(self.reset_states, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        self.reset_states()
        arch_run = getattr(self, arch); arch_run = tf.function(arch_run, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS); setattr(self, arch, arch_run)


    def metrics_spec(self):
        metrics_loss = OrderedDict()
        metrics_loss['1rewards*'] = {'-rewards_ma':np.float64, '-rewards_total+':np.float64, 'rewards_final=':np.float64}
        metrics_loss['1steps'] = {'steps+':np.int64}
        if arch == 'MU4':
            # metrics_loss['1rewards3*'] = {'-rewards_PGL_ma':np.float64, '-rewards_PGL_total+':np.float64, '-rewards_PGL_final=':np.float64}
            metrics_loss['1rewards2*'] = {'-rewards_PG_ma':np.float64, '-rewards_PG_total+':np.float64, '-rewards_PG_final=':np.float64}
            # metrics_loss['1extra'] = {'returns_pred':np.float64}
            metrics_loss['1netsACT*'] = {'-loss_act_ma':np.float64, '-loss_act':np.float64}
            metrics_loss['1netsPG*'] = {'-loss_PG_ma':np.float64, '-loss_PG':np.float64}
            # metrics_loss['1netsR'] = {'-loss_PG_snr':np.float64}; metrics_loss['1netsS'] = {'-loss_PG_std':np.float64}
            # metrics_loss['1nets8'] = {'loss_PGL':np.float64}; metrics_loss['1nets8S'] = {'-loss_PGL_std':np.float64}
            # metrics_loss['1nets6'] = {'loss_trans':np.float64}
            # metrics_loss['1nets2'] = {'loss_rwd':np.float64, 'loss_done':np.float64}
            # metrics_loss['1nets7'] = {'loss_trans_img':np.float64}
            # metrics_loss['1nets5'] = {'loss_rwd_img':np.float64, 'loss_done_img':np.float64}
            # metrics_loss['1nets7'] = {'loss_trans_dyn':np.float64}
            # metrics_loss['1nets5'] = {'loss_rwd_dyn':np.float64, 'loss_done_dyn':np.float64}
            # metrics_loss['1extra1'] = {'loss_actor_img':np.float64}
            # metrics_loss['1extra2'] = {'return_entropy_img':np.float64}
            # metrics_loss['1~extra3'] = {'-learn_rate':np.float64}
            # metrics_loss['1extra4'] = {'loss_meta':np.float64}
        if trader:
            metrics_loss['1rewards*'] = {'balance_avg':np.float64, 'balance_final=':np.float64}
            metrics_loss['1trader_marg*'] = {'equity':np.float64, 'margin_free':np.float64}
            metrics_loss['1trader_sim_time'] = {'sim_time_secs':np.float64}

        for loss_group in metrics_loss.values():
            for k in loss_group.keys():
                if k.endswith('=') or k.endswith('+'): loss_group[k] = [0 for i in range(max_episodes)]
                else: loss_group[k] = [[] for i in range(max_episodes)]
        self.metrics_loss = metrics_loss

    def metrics_update(self, *args):
        args = list(args)
        # for i in range(1,len(args)): args[i] = args[i].item()
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



    def gen_rnd(self):
        action = [None]*self.action_spec_len
        for i in range(self.action_spec_len):
            action[i] = tf.random.uniform((self.action_spec[i]['step_shape']), minval=self.action_spec[i]['min'], maxval=self.action_spec[i]['max'], dtype=self.action_spec[i]['dtype_out'])
        return action

    def gen_PG(self, inputs, use_img=False, store_real=False):
        action_logits = self.action(inputs, use_img=use_img, store_real=store_real)
        action = [None]*self.action_spec_len
        for i in range(self.action_spec_len):
            action_dist = self.action.dist[i](action_logits[i])
            action[i] = action_dist.sample()
        return action

    def gen_PGL(self, inputs, use_img=False, store_real=False):
        action_logits = self.actionL(inputs, use_img=use_img, store_real=store_real)
        action = [None]*self.action_spec_len
        for i in range(self.action_spec_len):
            action_dist = self.actionL.dist[i](action_logits[i])
            action[i] = action_dist.sample()
        return action

    def gen_act(self, inputs, use_img=False, store_real=False):
        action_logits = self.act(inputs, use_img=use_img, store_real=store_real)
        action = [None]*self.action_spec_len
        for i in range(self.action_spec_len):
            action_dist = self.act.dist[i](action_logits[i])
            action[i] = action_dist.sample()
        return action

    def train_PG(self, inputs, targets, returns, loss_value, store_memory=True, use_img=False, store_real=False):
        returns_calc = tf.cast(returns[0], self.compute_dtype)
        with tf.GradientTape() as tape_PG:
            action_logits = self.action(inputs, store_memory=store_memory, use_img=use_img, store_real=store_real)
            action_dist = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
            # loss_action = util.loss_PG(action_dist, action, returns)
            loss_action_lik = util.loss_likelihood(action_dist, targets)
            loss_action = loss_action_lik * returns_calc # _lEpA
            # loss_action = loss_action_lik * loss_value # _lEp9
            # loss_action = loss_action_lik * (returns_calc + loss_value) # _lEp5
            loss_action = loss_action * self.loss_scale
        gradients = tape_PG.gradient(loss_action, self.action.trainable_variables)
        for i in range(len(gradients)): gradients[i] = gradients[i] / self.loss_scale
        self.action.optimizer['action'].apply_gradients(zip(gradients, self.action.trainable_variables))
        return loss_action_lik

    # def train_PGL(self, inputs, targets, returns, loss_value, store_memory=True, use_img=False, store_real=False):
    #     returns_calc = tf.cast(returns[0], self.compute_dtype)
    #     with tf.GradientTape() as tape_PG:
    #         action_logits = self.actionL(inputs, store_memory=store_memory, use_img=use_img, store_real=store_real)
    #         action_dist = [None]*self.action_spec_len
    #         for i in range(self.action_spec_len): action_dist[i] = self.actionL.dist[i](action_logits[i])
    #         loss_action_lik = util.loss_likelihood(action_dist, targets)
    #         loss_action = loss_action_lik * (returns_calc + loss_value) # _lEp5
    #         loss_action = loss_action * self.loss_scale
    #     gradients = tape_PG.gradient(loss_action, self.actionL.trainable_variables)
    #     for i in range(len(gradients)): gradients[i] = gradients[i] / self.loss_scale
    #     self.actionL.optimizer['actionL'].apply_gradients(zip(gradients, self.actionL.trainable_variables))
    #     return loss_action_lik

    def train_act(self, inputs, targets, returns, store_memory=True, use_img=False, store_real=False):
        with tf.GradientTape() as tape_act:
            action_logits = self.act(inputs, store_memory=store_memory, use_img=use_img, store_real=store_real)
            action_dist = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action_dist[i] = self.act.dist[i](action_logits[i])
            loss_action = util.loss_likelihood(action_dist, targets)
            # loss_action = util.loss_PG(action_dist, targets, returns, returns_target=return_goal) # _lRt
            # loss_action = util.loss_PG(action_dist, targets, returns) # _lR
        gradients = tape_act.gradient(loss_action, self.act.trainable_variables)
        self.act.optimizer['act'].apply_gradients(zip(gradients, self.act.trainable_variables))
        return loss_action

    def train_trans(self, inputs, target_trans, target_reward, target_done, store_memory=True, training=True):
        with tf.GradientTape() as tape_trans:
        # with tf.GradientTape() as tape_trans, tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done:
            trans_logits = self.trans(inputs, store_memory=store_memory); trans_dist = self.trans.dist(trans_logits)
            latent_trans = trans_dist.sample()
            loss_tran = util.loss_likelihood(trans_dist, target_trans)
            # loss_tran = util.loss_diff(latent_trans, target_trans)
        gradients = tape_trans.gradient(loss_tran, self.trans.trainable_variables)
        self.trans.optimizer['trans'].apply_gradients(zip(gradients, self.trans.trainable_variables))

        with tf.GradientTape() as tape_reward:
        # with tape_reward:
            rwd_logits = self.rwd(latent_trans, store_memory=store_memory); rwd_dist = self.rwd.dist[0](rwd_logits[0])
            loss_reward = util.loss_likelihood(rwd_dist, target_reward)
        gradients = tape_reward.gradient(loss_reward, self.rwd.trainable_variables) # self.trans.trainable_variables + self.rwd.trainable_variables
        self.rwd.optimizer['rwd'].apply_gradients(zip(gradients, self.rwd.trainable_variables)) # self.trans.trainable_variables + self.rwd.trainable_variables

        with tf.GradientTape() as tape_done:
        # with tape_done:
            done_logits = self.done(latent_trans, store_memory=store_memory); done_dist = self.done.dist[0](done_logits[0])
            loss_done = util.loss_likelihood(done_dist, target_done)
        gradients = tape_done.gradient(loss_done, self.done.trainable_variables) # self.trans.trainable_variables + self.done.trainable_variables
        self.done.optimizer['done'].apply_gradients(zip(gradients, self.done.trainable_variables)) # self.trans.trainable_variables + self.done.trainable_variables

        return loss_tran, loss_reward, loss_done



    def MU4_img(self, inputs, gen):
        print("tracing -> GeneralAI MU4_img")
        obs = tf.TensorArray(self.latent_spec['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])
        actions = [None]*self.action_spec_len
        for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
        # rewards = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))
        returns = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))
        step_sizes = tf.TensorArray(tf.int32, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))
        entropies = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        step_size, step_scale = [self.step_size_one], 0
        step_loc = self.attn_img_step_locs[step_scale]

        inputs_step = {'obs':inputs['obs'], 'actions':self.action_zero_out, 'step_size':step_size, 'return_goal':inputs['return_goal']}
        step, dones = tf.constant(0), self.dones_zero
        while not dones[-1][0]:
            obs = obs.write(step, inputs_step['obs'])
            step_sizes = step_sizes.write(step, inputs_step['step_size'][0][0])

            # TODO condition actors by step_size?
            action = self.action_zero_out
            if gen == 0: action = self.gen_PG(inputs_step['obs'], use_img=True)
            if gen == 1: action = self.gen_act(inputs_step, use_img=True)
            # if gen == 2: action = self.gen_rnd()
            if gen == 2: action = self.gen_PGL(inputs_step['obs'], use_img=True)
            for i in range(self.action_spec_len): actions[i] = actions[i].write(step, action[i][-1])
            inputs_step['actions'] = action

            if step == step_loc:
                step_size = [tf.reshape(self.attn_img_step_sizesT[step_scale],(1,1))]
                step_scale += 1
                if step_scale != self.attn_img_scales: step_loc = self.attn_img_step_locs[step_scale]
                # if step != step_loc: step_size = self.step_size_one # TODO remove
            inputs_step['step_size'] = step_size

            trans_logits = self.trans(inputs_step, use_img=True); trans_dist = self.trans.dist(trans_logits)
            inputs_step['obs'] = trans_dist.sample()

            rwd_logits = self.rwd(inputs_step['obs'], use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
            done_logits = self.done(inputs_step['obs'], use_img=True); done_dist = self.done.dist[0](done_logits[0])
            reward, dones = tf.cast(rwd_dist.sample(), tf.float64), tf.cast(done_dist.sample(), tf.bool)
            if step_scale == self.attn_img_scales: dones = tf.constant([[True]])
            inputs_step['return_goal'] = [inputs_step['return_goal'][0] - reward]

            # entropy_rwd, entropy_done = rwd_dist.entropy(), done_dist.entropy()
            # entropy = (entropy_rwd + entropy_done) * 0.1
            # entropies = entropies.write(step, entropy)

            # rewards = rewards.write(step, reward[-1])
            returns = returns.write(step, [self.float64_zero])
            returns_updt = returns.stack()
            returns_updt = returns_updt + reward[-1]
            returns = returns.unstack(returns_updt)

            step += 1

        outputs = {}
        out_actions = [None]*self.action_spec_len
        for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
        outputs['obs'], outputs['actions'], outputs['returns'], outputs['step_size'], outputs['entropy'] = obs.stack(), out_actions, returns.stack(), step_sizes.stack(), entropies.stack()
        # outputs['rewards'] = rewards.stack()
        return outputs

    # def MU4_img_act_learn(self, inputs, action, training=True):
    #     print("tracing -> GeneralAI MU4_img_act_learn")

    #     self.act.reset_states(use_img=True)
    #     for step in tf.range(tf.shape(inputs['obs'])[0]):
    #         inputs_act = {'obs':inputs['obs'][step:step+1][0], 'return_goal':[inputs['returns'][step:step+1]], 'step_size':self.step_size_one}
    #         action_logits = self.act(inputs_act, use_img=True)

    #     inputs_act = {'obs':inputs['obs'][:1][0], 'return_goal':[inputs['returns'][:1]], 'step_size':self.step_size_one}
    #     with tf.GradientTape() as tape_action:
    #         action_logits = self.act(inputs_act, store_memory=False, use_img=True)
    #         action_dist = [None]*self.action_spec_len
    #         for i in range(self.action_spec_len): action_dist[i] = self.act.dist[i](action_logits[i])
    #         loss_action = util.loss_likelihood(action_dist, action)
    #     gradients = tape_action.gradient(loss_action, self.act.trainable_variables) # self.rep.trainable_variables
    #     self.act.optimizer['act'].apply_gradients(zip(gradients, self.act.trainable_variables)) # self.rep.trainable_variables
    #     # loss_actions = loss_actions.write(step, loss_action)

    def MU4_img_learner(self, inputs, gen, training=True):
        print("tracing -> GeneralAI MU4_img_learner")
        loss = {}
        loss_act = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_PG = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        episode_len = tf.shape(inputs['returns'])[0]
        for step in tf.range(episode_len):
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
            return_step = inputs['returns'][step:step+1]

            inputs_step = {'obs':inputs['obs'][step:step+1][0], 'step_size':[inputs['step_size'][step:step+1]], 'return_goal':[return_step]}
            with tf.GradientTape() as tape_act:
                action_logits = self.act(inputs_step, use_img=True)
                action_dist = [None]*self.action_spec_len
                for i in range(self.action_spec_len): action_dist[i] = self.act.dist[i](action_logits[i])
                loss_action = util.loss_likelihood(action_dist, action)
                # loss_action = util.loss_PG(action_dist, action, inputs['entropy'][step:step+1])
            gradients = tape_act.gradient(loss_action, self.act.trainable_variables)
            self.act.optimizer['act'].apply_gradients(zip(gradients, self.act.trainable_variables))
            loss_act = loss_act.write(step, loss_action)

            if gen == 0:
                with tf.GradientTape() as tape_PG:
                    action_logits = self.action(inputs_step['obs'], use_img=True)
                    action_dist = [None]*self.action_spec_len
                    for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
                    loss_action = util.loss_PG(action_dist, action, return_step)
                gradients = tape_PG.gradient(loss_action, self.action.trainable_variables)
                self.action.optimizer['action'].apply_gradients(zip(gradients, self.action.trainable_variables))
                loss_PG = loss_PG.write(step, loss_action)
            if gen == 2:
                with tf.GradientTape() as tape_PG:
                    action_logits = self.actionL(inputs_step['obs'], use_img=True)
                    action_dist = [None]*self.action_spec_len
                    for i in range(self.action_spec_len): action_dist[i] = self.actionL.dist[i](action_logits[i])
                    loss_action = util.loss_PG(action_dist, action, return_step)
                gradients = tape_PG.gradient(loss_action, self.actionL.trainable_variables)
                self.actionL.optimizer['actionL'].apply_gradients(zip(gradients, self.actionL.trainable_variables))
                loss_PG = loss_PG.write(step, loss_action)

        loss['act'], loss['PG'] = loss_act.concat(), loss_PG.concat()
        return loss


    def MU4_actor(self, inputs, gen, return_goal, return_goal_alt):
        print("tracing -> GeneralAI MU4_actor")
        loss = {}
        loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_trans = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_rewards = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_dones = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_trans_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_rewards_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_dones_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        # metric_returns_pred = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        # metric_entropy = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        obs, actions, returns = [None]*self.obs_spec_len, [None]*self.action_spec_len, [None]*self.attn_img_scales
        for i in range(self.obs_spec_len): obs[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['event_shape'])
        for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
        for i in range(self.attn_img_scales): returns[i] = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))
        rewards = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))
        dones = tf.TensorArray(tf.bool, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        # obs_rep = tf.TensorArray(self.latent_spec['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])
        # obs_trans = tf.TensorArray(self.latent_spec['dtype'], size=0, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])
        # obs_trans_img = tf.TensorArray(self.latent_spec['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])


        inputs_step = {'obs':self.latent_zero, 'step_size':[self.step_size_one], 'return_goal':[return_goal]}
        step, inputs['step'], dyn_img_obs = tf.constant(0), [self.step_zero], self.latent_zero

        rep_logits = self.rep(inputs); rep_dist = self.rep.dist(rep_logits)
        inputs_step['obs'] = rep_dist.sample()

        while not inputs['dones'][-1][0]:
            for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])

            # with tf.GradientTape(persistent=True) as tape_trans:
            # with tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done: # tf.GradientTape(persistent=True) as tape_action,
            # with tf.GradientTape(persistent=True) as tape_trans, tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done: # tf.GradientTape(persistent=True) as tape_action,
            # rep_logits = self.rep(inputs); rep_dist = self.rep.dist(rep_logits)
            # inputs_step['obs'] = rep_dist.sample()
            # obs_rep = obs_rep.write(step, inputs_step['obs'])

            # TODO change done to predict the number of steps until done
            # TODO train trans to match rep latent output (use cosine similarity if not using distribution latents)
            # TODO add transformer layer latents together for long horizon prediction and long history
            # TODO add ARS/MCTS/explorer here too (explorer is trained to find unique situations)
            # TODO train act on img but scale likelihood based on prediction entropy (higher entropy = flatter loss)
            # TODO scale likelihood based on action entropy (lower entropy = flatter loss)


            # TODO collect dyn model loss or entropy in the same fashion as returns ie future based, make actor who tries to maximize it
            # TODO can I use linear layers in the transformer to compress (4 to 1, 64 to 1, full episode to 1, etc) the far (past+future) latents?

            # ## _img
            # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
            # self.action.reset_states(use_img=True); self.act.reset_states(use_img=True)
            # outputs_img = self.MU4_img(inputs_step, gen)

            # self.action.reset_states(use_img=True); self.act.reset_states(use_img=True)
            # loss_act = self.MU4_img_learner(outputs_img, gen)
            # loss_actions = loss_actions.write(step, tf.expand_dims(tf.math.reduce_mean(loss_act['PG'], axis=0), axis=0))


            # ## _img2
            # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
            # outputs_img = self.MU4_img(inputs_step, 2)
            # self.action.reset_states(use_img=True); self.act.reset_states(use_img=True)
            # loss_act = self.MU4_img_learner(outputs_img, 2)

            # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
            # self.action.reset_states(use_img=True); self.act.reset_states(use_img=True)
            # outputs_img = self.MU4_img(inputs_step, 0)
            # # self.action.reset_states(use_img=True); self.act.reset_states(use_img=True)
            # # loss_act = self.MU4_img_learner(outputs_img, 0)
            # # loss_actions = loss_actions.write(step, tf.expand_dims(tf.math.reduce_mean(loss_act['PG'], axis=0), axis=0))
            # # action = self.gen_PG(inputs_step['obs'])
            # action = self.gen_PG(inputs_step['obs'], use_img=True, store_real=True) # _img

            # inputs_act = {'obs':inputs_step['obs'], 'step_size':[self.step_size_one], 'return_goal':[outputs_img['returns'][0:1]]}
            # with tf.GradientTape() as tape_act:
            #     action_logits = self.act(inputs_act, store_memory=False, use_img=True, store_real=True)
            #     action_dist = [None]*self.action_spec_len
            #     for i in range(self.action_spec_len): action_dist[i] = self.act.dist[i](action_logits[i])
            #     loss_action = util.loss_likelihood(action_dist, action)
            # gradients = tape_act.gradient(loss_action, self.act.trainable_variables)
            # self.act.optimizer['act'].apply_gradients(zip(gradients, self.act.trainable_variables))


            # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
            # self.action.reset_states(use_img=True); self.act.reset_states(use_img=True)
            # outputs_img = self.MU4_img(inputs_step, 1, return_goal_alt)
            # self.action.reset_states(use_img=True); self.act.reset_states(use_img=True)
            # loss_act = self.MU4_img_learner(outputs_img, 1)

            # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
            # self.action.reset_states(use_img=True); self.act.reset_states(use_img=True)
            # outputs_img = self.MU4_img(inputs_step, 1)
            # self.action.reset_states(use_img=True); self.act.reset_states(use_img=True)
            # loss_act = self.MU4_img_learner(outputs_img, 1)
            # loss_actions = loss_actions.write(step, tf.expand_dims(tf.math.reduce_mean(loss_act['act'], axis=0), axis=0))
            # action = self.gen_act(inputs_step, use_img=True, store_real=True) # _img



            action = self.action_zero_out
            if gen == 0:
                # action = self.gen_rnd()
                # action = self.gen_PG(inputs_step['obs'], use_img=True, store_real=True) # _img
                action = self.gen_PG(inputs_step['obs'])
            if gen == 1:
                # action = self.gen_act(inputs_step, use_img=True, store_real=True) # _img
                action = self.gen_act(inputs_step)
            if gen == 2:
                # action = self.gen_rnd()
                # action = self.gen_PGL(inputs_step['obs'], use_img=True, store_real=True) # _img
                action = self.gen_PGL(inputs_step['obs'])



            action_dis = [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                actions[i] = actions[i].write(step, action[i][-1])
                action_dis[i] = util.discretize(action[i], self.action_spec[i])

            np_in = tf.numpy_function(self.env_step, action_dis, self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs['obs'], inputs['rewards'], inputs['dones'] = np_in[:-2], np_in[-2], np_in[-1]
            # return_goal -= inputs['rewards']; return_goal_alt -= inputs['rewards']
            inputs_step['return_goal'] = [inputs_step['return_goal'][0] - inputs['rewards']]
            inputs['step'] = [tf.reshape(step+1,(1,1))]

            inputs_step_img = {'obs':dyn_img_obs, 'actions':action, 'step_size':[self.step_size_one]}
            inputs_step_dyn = {'obs':inputs_step['obs'], 'actions':action, 'step_size':[self.step_size_one]}
            rep_logits = self.rep(inputs); rep_dist = self.rep.dist(rep_logits)
            inputs_step['obs'] = rep_dist.sample()

            rewards = rewards.write(step, inputs['rewards'][-1])
            dones = dones.write(step, inputs['dones'][-1])

            for step_scale in range(self.attn_img_scales):
                returns[step_scale] = returns[step_scale].write(step, [self.float64_zero])
                returns_updt = returns[step_scale].stack()
                if step_scale != self.attn_img_scales-1:
                    step_size = self.attn_img_step_sizes[step_scale]
                    returns_temp = returns_updt[-step_size:] + inputs['rewards']
                    returns_updt = tf.concat([returns_updt[:-step_size], returns_temp], axis=0)
                else: returns_updt = returns_updt + inputs['rewards']
                returns[step_scale] = returns[step_scale].unstack(returns_updt)


            # trans_logits = self.trans(inputs_step_dyn); trans_dist = self.trans.dist(trans_logits)
            # latent_trans = trans_dist.sample()
            # rwd_logits = self.rwd(latent_trans); done_logits = self.done(latent_trans)

            # ## _dyn2 (imagination environment)
            # if step >= 1:
            #     self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
            #     with tf.GradientTape() as tape_trans_img:
            #     # with tf.GradientTape() as tape_trans_img, tf.GradientTape(persistent=True) as tape_reward_img, tf.GradientTape(persistent=True) as tape_done_img:
            #         trans_logits = self.trans(inputs_step_img, use_img=True); trans_dist = self.trans.dist(trans_logits)
            #         latent_trans_img = trans_dist.sample()
            #         loss_tran = util.loss_likelihood(trans_dist, inputs_step['obs'])
            #         # loss_tran = util.loss_diff(latent_trans_img, inputs_step['obs'])
            #     gradients = tape_trans_img.gradient(loss_tran, self.trans.trainable_variables)
            #     self.trans.optimizer['trans'].apply_gradients(zip(gradients, self.trans.trainable_variables))
            #     loss_trans_img = loss_trans_img.write(step, tf.expand_dims(tf.math.reduce_mean(loss_tran, axis=0), axis=0))

            #     with tf.GradientTape() as tape_reward_img:
            #     # with tape_reward_img:
            #         rwd_logits = self.rwd(latent_trans_img, use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
            #         loss_reward = util.loss_likelihood(rwd_dist, inputs['rewards'])
            #     gradients = tape_reward_img.gradient(loss_reward, self.rwd.trainable_variables) # self.trans.trainable_variables + self.rwd.trainable_variables
            #     self.rwd.optimizer['rwd'].apply_gradients(zip(gradients, self.rwd.trainable_variables)) # self.trans.trainable_variables + self.rwd.trainable_variables
            #     loss_rewards_img = loss_rewards_img.write(step, loss_reward)

            #     with tf.GradientTape() as tape_done_img:
            #     # with tape_done_img:
            #         done_logits = self.done(latent_trans_img, use_img=True); done_dist = self.done.dist[0](done_logits[0])
            #         loss_done = util.loss_likelihood(done_dist, inputs['dones'])
            #     gradients = tape_done_img.gradient(loss_done, self.done.trainable_variables) # self.trans.trainable_variables + self.done.trainable_variables
            #     self.done.optimizer['done'].apply_gradients(zip(gradients, self.done.trainable_variables)) # self.trans.trainable_variables + self.done.trainable_variables
            #     loss_dones_img = loss_dones_img.write(step, loss_done)


            # ## _dyn7 (real environment)
            # with tf.GradientTape() as tape_trans:
            # # with tf.GradientTape() as tape_trans, tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done:
            #     trans_logits = self.trans(inputs_step_dyn); trans_dist = self.trans.dist(trans_logits)
            #     dyn_img_obs = trans_dist.sample()
            #     loss_tran = util.loss_likelihood(trans_dist, inputs_step['obs'])
            #     # loss_tran = util.loss_diff(dyn_img_obs, inputs_step['obs'])
            # gradients = tape_trans.gradient(loss_tran, self.trans.trainable_variables) # self.rep.trainable_variables +
            # self.trans.optimizer['trans'].apply_gradients(zip(gradients, self.trans.trainable_variables)) # self.rep.trainable_variables +
            # loss_trans = loss_trans.write(step, tf.expand_dims(tf.math.reduce_mean(loss_tran, axis=0), axis=0))

            # with tf.GradientTape() as tape_reward:
            # # with tape_reward:
            #     rwd_logits = self.rwd(dyn_img_obs); rwd_dist = self.rwd.dist[0](rwd_logits[0])
            #     loss_reward = util.loss_likelihood(rwd_dist, inputs['rewards'])
            # gradients = tape_reward.gradient(loss_reward, self.rwd.trainable_variables) # self.rep.trainable_variables + self.trans.trainable_variables +
            # self.rwd.optimizer['rwd'].apply_gradients(zip(gradients, self.rwd.trainable_variables)) # self.rep.trainable_variables + self.trans.trainable_variables +
            # loss_rewards = loss_rewards.write(step, loss_reward)

            # with tf.GradientTape() as tape_done:
            # # with tape_done:
            #     done_logits = self.done(dyn_img_obs); done_dist = self.done.dist[0](done_logits[0])
            #     loss_done = util.loss_likelihood(done_dist, inputs['dones'])
            # gradients = tape_done.gradient(loss_done, self.done.trainable_variables) # self.rep.trainable_variables + self.trans.trainable_variables +
            # self.done.optimizer['done'].apply_gradients(zip(gradients, self.done.trainable_variables)) # self.rep.trainable_variables + self.trans.trainable_variables +
            # loss_dones = loss_dones.write(step, loss_done)



            step += 1
        for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])

        outputs = {}
        out_obs, out_actions, out_returns = [None]*self.obs_spec_len, [None]*self.action_spec_len, [None]*self.attn_img_scales
        for i in range(self.obs_spec_len): out_obs[i] = obs[i].stack()
        for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
        for i in range(self.attn_img_scales): out_returns[i] = returns[i].stack()
        outputs['obs'], outputs['actions'], outputs['returns'], outputs['rewards'], outputs['dones'] = out_obs, out_actions, out_returns, rewards.stack(), dones.stack()
        # outputs['obs_rep'], outputs['obs_trans'], outputs['obs_trans_img'] = obs_rep.stack(), obs_trans.stack(), obs_trans_img.stack()

        loss['action'], loss['trans'], loss['reward'], loss['done'] = loss_actions.concat(), loss_trans.concat(), loss_rewards.concat(), loss_dones.concat()
        loss['trans_img'], loss['reward_img'], loss['done_img'] = loss_trans_img.concat(), loss_rewards_img.concat(), loss_dones_img.concat()
        # loss['entropy'], loss['returns_pred'] = metric_entropy.concat(), metric_returns_pred.concat()
        return outputs, inputs, loss


    def MU4_rep_learner(self, inputs, gen, training=True):
        print("tracing -> GeneralAI MU4_rep_learner")
        loss = {}
        loss_act = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_PG = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_trans = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_rewards = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_dones = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        episode_len = tf.shape(inputs['dones'])[0]
        for step in tf.range(episode_len):
            obs = [None]*self.obs_spec_len
            for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
            return_step = inputs['returns'][-1][step:step+1]

            inputs_step = {'obs':obs, 'actions':action, 'step':[tf.reshape(step,(1,1))], 'step_size':[self.step_size_one], 'return_goal':[return_step]}
            with tf.GradientTape(persistent=True) as tape_act, tf.GradientTape(persistent=True) as tape_PG, tf.GradientTape(persistent=True) as tape_trans, tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done:
                rep_logits = self.rep(inputs_step); rep_dist = self.rep.dist(rep_logits)
                inputs_step['obs'] = rep_dist.sample()

            with tape_act:
                action_logits = self.act(inputs_step)
                action_dist = [None]*self.action_spec_len
                for i in range(self.action_spec_len): action_dist[i] = self.act.dist[i](action_logits[i])
                loss_action = util.loss_likelihood(action_dist, action)
            gradients = tape_act.gradient(loss_action, self.rep.trainable_variables) # + self.act.trainable_variables
            self.rep.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables)) # + self.act.trainable_variables
            loss_act = loss_act.write(step, loss_action)

            if gen == 0:
                with tape_PG:
                    action_logits = self.action(inputs_step['obs'])
                    action_dist = [None]*self.action_spec_len
                    for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
                    # loss_action = util.loss_PG(action_dist, action, return_step)
                    loss_action = util.loss_likelihood(action_dist, action)
                    loss_action = loss_action * self.loss_scale
                gradients = tape_PG.gradient(loss_action, self.rep.trainable_variables) # + self.action.trainable_variables
                for i in range(len(gradients)): gradients[i] = gradients[i] / self.loss_scale
                self.rep.optimizer['PG'].apply_gradients(zip(gradients, self.rep.trainable_variables)) # + self.action.trainable_variables
                loss_PG = loss_PG.write(step, loss_action)
            if gen == 2:
                with tape_PG:
                    action_logits = self.actionL(inputs_step['obs'])
                    action_dist = [None]*self.action_spec_len
                    for i in range(self.action_spec_len): action_dist[i] = self.actionL.dist[i](action_logits[i])
                    # loss_action = util.loss_PG(action_dist, action, return_step)
                    loss_action = util.loss_likelihood(action_dist, action)
                    loss_action = loss_action * self.loss_scale
                gradients = tape_PG.gradient(loss_action, self.rep.trainable_variables) # + self.actionL.trainable_variables
                for i in range(len(gradients)): gradients[i] = gradients[i] / self.loss_scale
                self.rep.optimizer['PGL'].apply_gradients(zip(gradients, self.rep.trainable_variables)) # + self.actionL.trainable_variables
                loss_PG = loss_PG.write(step, loss_action)

            # obs = [None]*self.obs_spec_len
            # for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step+1:step+2]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
            # inputs_step_next = {'obs':obs, 'step':[tf.reshape(step+1,(1,1))], 'step_size':[self.step_size_one]}
            # rep_logits = self.rep(inputs_step_next, store_memory=False); rep_dist = self.rep.dist(rep_logits)
            # rep_target = rep_dist.sample()

            # with tape_trans, tape_reward, tape_done:
            #     trans_logits = self.trans(inputs_step); trans_dist = self.trans.dist(trans_logits)
            #     inputs_step['obs'] = trans_dist.sample()
            #     loss_tran = util.loss_likelihood(trans_dist, rep_target) # / rep_target.shape[-1]
            #     # loss_tran = util.loss_diff(inputs_step['obs'], rep_target)
            # gradients = tape_trans.gradient(loss_tran, self.rep.trainable_variables)
            # self.rep.optimizer['trans'].apply_gradients(zip(gradients, self.rep.trainable_variables)) # + self.trans.trainable_variables
            # loss_trans = loss_trans.write(step, tf.expand_dims(tf.math.reduce_mean(loss_tran, axis=0), axis=0))

            # with tape_reward:
            #     rwd_logits = self.rwd(inputs_step['obs']); rwd_dist = self.rwd.dist[0](rwd_logits[0])
            #     loss_reward = util.loss_likelihood(rwd_dist, inputs['rewards'][step:step+1])
            # gradients = tape_reward.gradient(loss_reward, self.rep.trainable_variables) # + self.trans.trainable_variables + self.rwd.trainable_variables
            # self.rep.optimizer['rwd'].apply_gradients(zip(gradients, self.rep.trainable_variables)) # + self.trans.trainable_variables + self.rwd.trainable_variables
            # loss_rewards = loss_rewards.write(step, loss_reward)

            # with tape_done:
            #     done_logits = self.done(inputs_step['obs']); done_dist = self.done.dist[0](done_logits[0])
            #     loss_done = util.loss_likelihood(done_dist, inputs['dones'][step:step+1])
            # gradients = tape_done.gradient(loss_done, self.rep.trainable_variables) # + self.trans.trainable_variables + self.done.trainable_variables
            # self.rep.optimizer['done'].apply_gradients(zip(gradients, self.rep.trainable_variables)) # + self.trans.trainable_variables + self.done.trainable_variables
            # loss_dones = loss_dones.write(step, loss_done)

        loss['act'], loss['PG'] = loss_act.concat(), loss_PG.concat()
        loss['trans'], loss['reward'], loss['done'] = loss_trans.concat(), loss_rewards.concat(), loss_dones.concat()
        return loss

    def MU4_rep_returns(self, inputs, training=True):
        print("tracing -> GeneralAI MU4_rep_returns")
        obs_rep = tf.TensorArray(self.latent_spec['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])
        obs_rep_ret = [None]*self.attn_img_scales
        for i in range(self.attn_img_scales): obs_rep_ret[i] = tf.TensorArray(self.latent_spec['dtype'], size=0, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])

        episode_len = tf.shape(inputs['dones'])[0]
        for step in tf.range(episode_len+1):
            obs = [None]*self.obs_spec_len
            for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])

            inputs_step = {'obs':obs, 'step':[tf.reshape(step,(1,1))]}
            rep_logits = self.rep(inputs_step); rep_dist = self.rep.dist(rep_logits)
            latent_rep = rep_dist.sample()
            obs_rep = obs_rep.write(step, latent_rep)

            for step_scale in range(self.attn_img_scales):
                obs_rep_ret[step_scale] = obs_rep_ret[step_scale].write(step, self.latent_zero)
                returns_updt = obs_rep_ret[step_scale].stack()
                if step_scale != self.attn_img_scales-1:
                    step_size = self.attn_img_step_sizes[step_scale]
                    returns_temp = returns_updt[-step_size:] + latent_rep
                    returns_updt = tf.concat([returns_updt[:-step_size], returns_temp], axis=0)
                else: returns_updt = returns_updt + latent_rep
                obs_rep_ret[step_scale] = obs_rep_ret[step_scale].unstack(returns_updt)

        outputs = {}
        out_obs_rep_ret = [None]*self.attn_img_scales
        for i in range(self.attn_img_scales): out_obs_rep_ret[i] = obs_rep_ret[i].stack()
        outputs['obs_rep'], outputs['obs_rep_ret'] = obs_rep.stack(), out_obs_rep_ret
        return outputs

    def MU4_act_PG_learner(self, inputs, inputs_rep, gen, return_goal, training=True):
        print("tracing -> GeneralAI MU4_act_PG_learner")
        loss = {}
        loss_act = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_PG = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_act_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_PG_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        episode_len = tf.shape(inputs['dones'])[0]
        for step in tf.range(episode_len):
            # obs = [None]*self.obs_spec_len
            # for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
            return_step = inputs['returns'][-1][step:step+1]

            # inputs_step = {'obs':obs, 'step':[tf.reshape(step,(1,1))], 'step_size':[self.step_size_one], 'return_goal':[return_step]}
            # rep_logits = self.rep(inputs_step); rep_dist = self.rep.dist(rep_logits)
            # inputs_step['obs'] = rep_dist.sample()

            inputs_step = {'obs':inputs_rep['obs_rep'][step:step+1][0], 'step_size':[self.step_size_one], 'return_goal':[return_step]}

            # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True) # _img
            # self.act.reset_states(use_img=True)
            # outputs_img = self.MU4_img(inputs_step, 1)
            # loss_action = self.train_act(inputs_step, action, return_step, store_memory=False, use_img=True, store_real=True)
            # loss_act_img = loss_act_img.write(step, loss_action)

            loss_action = self.train_act(inputs_step, action, return_step)
            loss_act = loss_act.write(step, loss_action)

            if gen == 0:
                # inputs_step_dyn = {'obs':inputs_rep['obs_rep'][step:step+1][0], 'actions':action, 'step_size':[self.step_size_max]}
                # trans_logits = self.trans(inputs_step_dyn, store_memory=False); trans_dist = self.trans.dist(trans_logits)
                # latent_trans = trans_dist.sample()
                # rwd_logits = self.rwd(latent_trans, store_memory=False); rwd_dist = self.rwd.dist[0](rwd_logits[0])
                # loss_value = util.loss_likelihood(rwd_dist, return_step)

                # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True) # _img
                # self.action.reset_states(use_img=True)
                # outputs_img = self.MU4_img(inputs_step, 0)
                # loss_action = self.train_PG(inputs_step['obs'], action, return_step, loss_value, store_memory=False, use_img=True, store_real=True)
                # loss_PG_img = loss_PG_img.write(step, loss_action)

                loss_action = self.train_PG(inputs_step['obs'], action, return_step, None)
                loss_PG = loss_PG.write(step, loss_action)

            # if gen == 2:
            #     self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True) # _img
            #     self.actionL.reset_states(use_img=True)
            #     outputs_img = self.MU4_img(inputs_step, 0)
            #     loss_action = self.train_PGL(inputs_step['obs'], action, return_step, loss_value, store_memory=False, use_img=True, store_real=True)
            #     loss_PG_img = loss_PG_img.write(step, loss_action)

            #     loss_action = self.train_PGL(inputs_step['obs'], action, return_step, loss_value)
            #     loss_PG = loss_PG.write(step, loss_action)

            # inputs_step_store = {'obs':inputs_step['obs'], 'actions':action, 'step_size':[self.step_size_one]}
            # trans_logits = self.trans(inputs_step_store); trans_dist = self.trans.dist(trans_logits)
            # latent_trans = trans_dist.sample()
            # rwd_logits = self.rwd(latent_trans); done_logits = self.done(latent_trans)

        loss['act'], loss['PG'] = loss_act.concat(), loss_PG.concat()
        # loss['act_img'], loss['PG_img'] = loss_act_img.concat(), loss_PG_img.concat()
        return loss

    # def MU4_act_learner(self, inputs, return_goal, training=True):
    #     print("tracing -> GeneralAI MU4_act_learner")
    #     loss = {}
    #     loss_act = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #     episode_len = tf.shape(inputs['dones'])[0]
    #     for step in tf.range(episode_len):
    #         obs = [None]*self.obs_spec_len
    #         for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
    #         action = [None]*self.action_spec_len
    #         for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
    #         return_step = inputs['returns'][-1][step:step+1]

    #         inputs_step = {'obs':obs, 'step_size':self.step_size_one}
    #         rep_logits = self.rep(inputs_step, step=step); rep_dist = self.rep.dist(rep_logits)
    #         inputs_step['obs'] = rep_dist.sample()

    #         inputs_act = {'obs':inputs_step['obs'], 'return_goal':return_step, 'step_size':self.step_size_one}
    #         with tf.GradientTape() as tape_act:
    #             action_logits = self.act(inputs_act)
    #             action_dist = [None]*self.action_spec_len
    #             for i in range(self.action_spec_len): action_dist[i] = self.act.dist[i](action_logits[i])
    #             loss_action = util.loss_likelihood(action_dist, action)
    #             # loss_action = util.loss_PG(action_dist, action, return_step, returns_target=return_goal) # _lRt
    #             # loss_action = util.loss_PG(action_dist, action, return_step) # _lR
    #         gradients = tape_act.gradient(loss_action, self.act.trainable_variables)
    #         self.act.optimizer['act'].apply_gradients(zip(gradients, self.act.trainable_variables))
    #         loss_act = loss_act.write(step, loss_action)
    #         # return_goal -= inputs['rewards'][step:step+1]; return_goal.set_shape((1,1))

    #     loss['act'] = loss_act.concat()
    #     return loss

    # def MU4_PG_learner(self, inputs, training=True):
    #     print("tracing -> GeneralAI MU4_PG_learner")
    #     loss = {}
    #     loss_PG = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #     episode_len = tf.shape(inputs['dones'])[0]
    #     for step in tf.range(episode_len):
    #         obs = [None]*self.obs_spec_len
    #         for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
    #         action = [None]*self.action_spec_len
    #         for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
    #         return_step = inputs['returns'][-1][step:step+1]

    #         inputs_step = {'obs':obs, 'step_size':self.step_size_one}
    #         rep_logits = self.rep(inputs_step, step=step); rep_dist = self.rep.dist(rep_logits)
    #         inputs_step['obs'] = rep_dist.sample()

    #         with tf.GradientTape() as tape_PG:
    #             action_logits = self.action(inputs_step)
    #             action_dist = [None]*self.action_spec_len
    #             for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
    #             loss_action = util.loss_PG(action_dist, action, return_step)
    #         gradients = tape_PG.gradient(loss_action, self.action.trainable_variables)
    #         self.action.optimizer['action'].apply_gradients(zip(gradients, self.action.trainable_variables))
    #         loss_PG = loss_PG.write(step, loss_action)

    #     loss['PG'] = loss_PG.concat()
    #     return loss

    def MU4_dyn_learner4(self, inputs, inputs_rep, training=True):
        print("tracing -> GeneralAI MU4_dyn_learner4")
        loss = {}
        loss_trans = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_rewards = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_dones = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        episode_len = tf.shape(inputs['dones'])[0]
        for step in tf.range(episode_len):
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])

            step_size_max = episode_len - step
            inputs_step_ret = {'obs':inputs_rep['obs_rep'][step:step+1][0], 'actions':action, 'step_size':[self.step_size_one]}
            for step_scale in range(self.attn_img_scales):
                step_size, target_done_ret = self.attn_img_step_sizes[step_scale], tf.constant([[False]])
                if step_size >= step_size_max: step_size, target_done_ret = step_size_max, tf.constant([[True]])
                if step_scale == self.attn_img_scales-1: target_done_ret = tf.constant([[True]])
                target_trans_ret, target_reward_ret = inputs_rep['obs_rep_ret'][step_scale][step+1:step+2][0], inputs['returns'][step_scale][step:step+1]

                inputs_step_ret['step_size'] = [tf.reshape(step_size,(1,1))]
                if step >= self.attn_img_step_locs[step_scale]:
                    loss_tran_ret, loss_reward_ret, loss_done_ret = self.train_trans(inputs_step_ret, target_trans_ret, target_reward_ret, target_done_ret, store_memory=False) # train current to next step size (1->4)
                    # loss_trans_ret = loss_trans_ret.write(step, tf.expand_dims(tf.math.reduce_mean(loss_tran_ret, axis=0), axis=0))
                    # loss_rewards_ret = loss_rewards_ret.write(step, loss_reward_ret)
                    # loss_dones_ret = loss_dones_ret.write(step, loss_done_ret)

                inputs_step_ret['obs'] = inputs_rep['obs_rep_ret'][step_scale][step:step+1][0]
                if step > self.attn_img_step_locs[step_scale]:
                    loss_tran_ret, loss_reward_ret, loss_done_ret = self.train_trans(inputs_step_ret, target_trans_ret, target_reward_ret, target_done_ret, store_memory=False) # train next to next step size (4->4)
                    # loss_trans_ret = loss_trans_ret.write(step, tf.expand_dims(tf.math.reduce_mean(loss_tran_ret, axis=0), axis=0))
                    # loss_rewards_ret = loss_rewards_ret.write(step, loss_reward_ret)
                    # loss_dones_ret = loss_dones_ret.write(step, loss_done_ret)

            # TODO always train 'step_size':[self.step_size_max] ?

            inputs_step_store = {'obs':inputs_rep['obs_rep'][step:step+1][0], 'actions':action, 'step_size':[self.step_size_one]}
            # trans_logits = self.trans(inputs_step_store); trans_dist = self.trans.dist(trans_logits)
            # latent_trans = trans_dist.sample()
            # rwd_logits = self.rwd(latent_trans); done_logits = self.done(latent_trans)
            target_trans, target_reward, target_done = inputs_rep['obs_rep'][step+1:step+2][0], inputs['rewards'][step:step+1], inputs['dones'][step:step+1]
            loss_tran, loss_reward, loss_done = self.train_trans(inputs_step_store, target_trans, target_reward, target_done)
            loss_trans = loss_trans.write(step, tf.expand_dims(tf.math.reduce_mean(loss_tran, axis=0), axis=0))
            loss_rewards = loss_rewards.write(step, loss_reward)
            loss_dones = loss_dones.write(step, loss_done)

        loss['trans'], loss['reward'], loss['done'] = loss_trans.concat(), loss_rewards.concat(), loss_dones.concat()
        return loss

    # def MU4_dyn_learner3(self, inputs, training=True):
    #     print("tracing -> GeneralAI MU4_dyn_learner3")
    #     loss = {}
    #     loss_trans_ret = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_rewards_ret = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_dones_ret = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #     episode_len = tf.shape(inputs['dones'])[0]

    #     # TODO put this in seperate function and use for both dyn and agent training
    #     obs_rep = tf.TensorArray(self.latent_spec['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])
    #     obs_rep_ret = [None]*self.attn_img_scales
    #     for i in range(self.attn_img_scales): obs_rep_ret[i] = tf.TensorArray(self.latent_spec['dtype'], size=0, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])

    #     for step in tf.range(episode_len+1):
    #         obs = [None]*self.obs_spec_len
    #         for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])

    #         inputs_step = {'obs':obs, 'step':[tf.reshape(step,(1,1))]}
    #         rep_logits = self.rep(inputs_step); rep_dist = self.rep.dist(rep_logits)
    #         latent_rep = rep_dist.sample()
    #         obs_rep = obs_rep.write(step, latent_rep)

    #         for step_scale in range(self.attn_img_scales):
    #             obs_rep_ret[step_scale] = obs_rep_ret[step_scale].write(step, self.latent_zero)
    #             returns_updt = obs_rep_ret[step_scale].stack()
    #             if step_scale != self.attn_img_scales-1:
    #                 step_size = self.attn_img_step_sizes[step_scale]
    #                 returns_temp = returns_updt[-step_size:] + latent_rep
    #                 returns_updt = tf.concat([returns_updt[:-step_size], returns_temp], axis=0)
    #             else: returns_updt = returns_updt + latent_rep
    #             obs_rep_ret[step_scale] = obs_rep_ret[step_scale].unstack(returns_updt)

    #     out_obs_rep_ret = [None]*self.attn_img_scales
    #     for i in range(self.attn_img_scales): out_obs_rep_ret[i] = obs_rep_ret[i].stack()
    #     inputs['obs_rep'], inputs['obs_rep_ret'] = obs_rep.stack(), out_obs_rep_ret

    #     for step in tf.range(episode_len):
    #         action = [None]*self.action_spec_len
    #         for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])

    #         step_size_max = episode_len - step
    #         inputs_step_ret = {'obs':inputs['obs_rep'][step:step+1][0], 'actions':action, 'step_size':[self.step_size_one]}
    #         for step_scale in range(self.attn_img_scales):
    #             step_size, target_done = self.attn_img_step_sizes[step_scale], tf.constant([[False]])
    #             if step_size >= step_size_max: step_size, target_done = step_size_max, tf.constant([[True]])
    #             if step_scale == self.attn_img_scales-1: target_done = tf.constant([[True]])
    #             target_trans, target_reward = inputs['obs_rep_ret'][step_scale][step+1:step+2][0], inputs['returns'][step_scale][step:step+1]

    #             inputs_step_ret['step_size'] = [tf.reshape(step_size,(1,1))]
    #             loss_tran, loss_reward, loss_done = self.train_trans(inputs_step_ret, target_trans, target_reward, target_done, store_memory=False) # train current to next step size (1->4)
    #             loss_trans_ret = loss_trans_ret.write(step, tf.expand_dims(tf.math.reduce_mean(loss_tran, axis=0), axis=0))
    #             loss_rewards_ret = loss_rewards_ret.write(step, loss_reward)
    #             loss_dones_ret = loss_dones_ret.write(step, loss_done)

    #             inputs_step_ret['obs'] = inputs['obs_rep_ret'][step_scale][step:step+1][0]
    #             loss_tran, loss_reward, loss_done = self.train_trans(inputs_step_ret, target_trans, target_reward, target_done, store_memory=False) # train next to next step size (4->4)
    #             loss_trans_ret = loss_trans_ret.write(step, tf.expand_dims(tf.math.reduce_mean(loss_tran, axis=0), axis=0))
    #             loss_rewards_ret = loss_rewards_ret.write(step, loss_reward)
    #             loss_dones_ret = loss_dones_ret.write(step, loss_done)

    #         inputs_step_store = {'obs':inputs['obs_rep'][step:step+1][0], 'actions':action, 'step_size':[self.step_size_one]}
    #         trans_logits = self.trans(inputs_step_store); trans_dist = self.trans.dist(trans_logits)
    #         latent_trans = trans_dist.sample()
    #         rwd_logits = self.rwd(latent_trans); done_logits = self.done(latent_trans)

    #     loss['trans'], loss['reward'], loss['done'] = loss_trans_ret.concat(), loss_rewards_ret.concat(), loss_dones_ret.concat()
    #     return loss

    # def MU4_dyn_learner2(self, inputs, gen, training=True):
    #     print("tracing -> GeneralAI MU4_dyn_learner2")
    #     loss = {}
    #     loss_PG_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_act_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_trans_ret = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_rewards_ret = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_dones_ret = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_trans = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_rewards = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_dones = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_PG = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_act = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #     episode_len = tf.shape(inputs['dones'])[0]


    #     obs_rep = tf.TensorArray(self.latent_spec['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])
    #     obs_rep_ret = [None]*self.attn_img_scales
    #     for i in range(self.attn_img_scales): obs_rep_ret[i] = tf.TensorArray(self.latent_spec['dtype'], size=0, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])

    #     for step in tf.range(episode_len+1):
    #         obs = [None]*self.obs_spec_len
    #         for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])

    #         inputs_step = {'obs':obs}
    #         rep_logits = self.rep(inputs_step, step=step); rep_dist = self.rep.dist(rep_logits)
    #         inputs_step['obs'] = rep_dist.sample()

    #         obs_rep = obs_rep.write(step, inputs_step['obs'])

    #         obs_rep_ret[-1] = obs_rep_ret[-1].write(step, self.latent_zero)
    #         returns_updt = obs_rep_ret[-1].stack()
    #         returns_updt = returns_updt + inputs_step['obs']
    #         obs_rep_ret[-1] = obs_rep_ret[-1].unstack(returns_updt)

    #     out_obs_rep_ret = [None]*self.attn_img_scales
    #     for i in range(self.attn_img_scales): out_obs_rep_ret[i] = obs_rep_ret[i].stack()
    #     inputs['obs_rep'], inputs['obs_rep_ret'] = obs_rep.stack(), out_obs_rep_ret


    #     for step in tf.range(episode_len):
    #         inputs_step = {'obs':inputs['obs_rep'][step:step+1][0], 'actions':self.action_zero_out, 'step_size':self.step_size_one}


    #         loss_PG_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_act_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_trans_ret_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_rewards_ret_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_dones_ret_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_trans_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_rewards_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_dones_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #         self.action.reset_states(use_img=True); self.act.reset_states(use_img=True)
    #         self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
    #         inputs_step_img = {'obs':inputs_step['obs'], 'actions':self.action_zero_out, 'step_size':self.step_size_one}
    #         for step_img in tf.range(step, episode_len):
    #             action = [None]*self.action_spec_len
    #             for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step_img:step_img+1]; action[i].set_shape(self.action_spec[i]['step_shape'])


    #             # ## _img
    #             # return_step = inputs['returns'][-1][step_img:step_img+1]
    #             # if gen == 0:
    #             #     with tf.GradientTape() as tape_PG:
    #             #         action_logits = self.action(inputs_step_img, use_img=True)
    #             #         action_dist = [None]*self.action_spec_len
    #             #         for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
    #             #         loss_action = util.loss_PG(action_dist, action, return_step)
    #             #     gradients = tape_PG.gradient(loss_action, self.action.trainable_variables)
    #             #     self.action.optimizer['action'].apply_gradients(zip(gradients, self.action.trainable_variables))
    #             #     loss_PG_img_accu = loss_PG_img_accu.write(step_img, loss_action)

    #             # inputs_act = {'obs':inputs_step_img['obs'], 'return_goal':return_step, 'step_size':self.step_size_one}
    #             # with tf.GradientTape() as tape_act_img:
    #             #     action_logits = self.act(inputs_act, use_img=True)
    #             #     action_dist = [None]*self.action_spec_len
    #             #     for i in range(self.action_spec_len): action_dist[i] = self.act.dist[i](action_logits[i])
    #             #     loss_action = util.loss_likelihood(action_dist, action)
    #             # gradients = tape_act_img.gradient(loss_action, self.act.trainable_variables)
    #             # self.act.optimizer['act'].apply_gradients(zip(gradients, self.act.trainable_variables))
    #             # loss_act_img_accu = loss_act_img_accu.write(step_img, loss_action)


    #             step_scale, step_size_max, step_size = -1, episode_len - step_img, self.attn_img_step_sizes[-1]
    #             step_size = step_size_max if step_size >= step_size_max else step_size
    #             inputs_step_img_ret = {'obs':inputs_step_img['obs'], 'actions':action, 'step_size':step_size}

    #             with tf.GradientTape() as tape_trans, tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done:
    #                 trans_logits = self.trans(inputs_step_img_ret, store_memory=False, use_img=True); trans_dist = self.trans.dist(trans_logits)
    #                 inputs_step_img_ret['obs'] = trans_dist.sample()
    #                 loss_tran = util.loss_likelihood(trans_dist, inputs['obs_rep_ret'][step_scale][step_img+1:step_img+2][0])
    #                 # loss_tran = util.loss_diff(inputs_step_img_ret['obs'], inputs['obs_rep_ret'][step_scale][step_img+1:step_img+2][0])
    #             gradients = tape_trans.gradient(loss_tran, self.trans.trainable_variables)
    #             self.trans.optimizer['trans'].apply_gradients(zip(gradients, self.trans.trainable_variables))
    #             loss_trans_ret_accu = loss_trans_ret_accu.write(step, tf.expand_dims(tf.math.reduce_mean(loss_tran, axis=0), axis=0))

    #             with tape_reward:
    #                 rwd_logits = self.rwd(inputs_step_img_ret, store_memory=False, use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #                 loss_reward = util.loss_likelihood(rwd_dist, inputs['returns'][step_scale][step_img:step_img+1])
    #             gradients = tape_reward.gradient(loss_reward, self.trans.trainable_variables + self.rwd.trainable_variables) # + self.rwd.trainable_variables
    #             self.rwd.optimizer['rwd'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.rwd.trainable_variables)) # + self.rwd.trainable_variables
    #             loss_rewards_ret_accu = loss_rewards_ret_accu.write(step_img, loss_reward)

    #             with tape_done:
    #                 done_logits = self.done(inputs_step_img_ret, store_memory=False, use_img=True); done_dist = self.done.dist[0](done_logits[0])
    #                 loss_done = util.loss_likelihood(done_dist, tf.constant([[True]]))
    #             gradients = tape_done.gradient(loss_done, self.trans.trainable_variables + self.done.trainable_variables) # + self.done.trainable_variables
    #             self.done.optimizer['done'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.done.trainable_variables)) # + self.done.trainable_variables
    #             loss_dones_ret_accu = loss_dones_ret_accu.write(step_img, loss_done)


    #             inputs_step_img['actions'] = action
    #             with tf.GradientTape() as tape_trans, tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done:
    #                 trans_logits = self.trans(inputs_step_img, use_img=True); trans_dist = self.trans.dist(trans_logits)
    #                 inputs_step_img['obs'] = trans_dist.sample()
    #                 loss_tran = util.loss_likelihood(trans_dist, inputs['obs_rep'][step_img+1:step_img+2][0])
    #                 # loss_tran = util.loss_diff(inputs_step_img['obs'], inputs['obs_rep'][step_img+1:step_img+2][0])
    #             gradients = tape_trans.gradient(loss_tran, self.trans.trainable_variables)
    #             self.trans.optimizer['trans'].apply_gradients(zip(gradients, self.trans.trainable_variables))
    #             loss_trans_accu = loss_trans_accu.write(step, tf.expand_dims(tf.math.reduce_mean(loss_tran, axis=0), axis=0))

    #             with tape_reward:
    #                 rwd_logits = self.rwd(inputs_step_img, use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #                 loss_reward = util.loss_likelihood(rwd_dist, inputs['rewards'][step_img:step_img+1])
    #             gradients = tape_reward.gradient(loss_reward, self.trans.trainable_variables + self.rwd.trainable_variables) # + self.rwd.trainable_variables
    #             self.rwd.optimizer['rwd'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.rwd.trainable_variables)) # + self.rwd.trainable_variables
    #             loss_rewards_accu = loss_rewards_accu.write(step_img, loss_reward)

    #             with tape_done:
    #                 done_logits = self.done(inputs_step_img, use_img=True); done_dist = self.done.dist[0](done_logits[0])
    #                 loss_done = util.loss_likelihood(done_dist, inputs['dones'][step_img:step_img+1])
    #             gradients = tape_done.gradient(loss_done, self.trans.trainable_variables + self.done.trainable_variables) # + self.done.trainable_variables
    #             self.done.optimizer['done'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.done.trainable_variables)) # + self.done.trainable_variables
    #             loss_dones_accu = loss_dones_accu.write(step_img, loss_done)

    #         loss_PG_img = loss_PG_img.write(step, tf.math.reduce_mean(loss_PG_img_accu.stack(), axis=0))
    #         loss_act_img = loss_act_img.write(step, tf.math.reduce_mean(loss_act_img_accu.stack(), axis=0))
    #         loss_trans_ret = loss_trans_ret.write(step, tf.math.reduce_mean(loss_trans_ret_accu.stack(), axis=0))
    #         loss_rewards_ret = loss_rewards_ret.write(step, tf.math.reduce_mean(loss_rewards_ret_accu.stack(), axis=0))
    #         loss_dones_ret = loss_dones_ret.write(step, tf.math.reduce_mean(loss_dones_ret_accu.stack(), axis=0))
    #         loss_trans = loss_trans.write(step, tf.math.reduce_mean(loss_trans_accu.stack(), axis=0))
    #         loss_rewards = loss_rewards.write(step, tf.math.reduce_mean(loss_rewards_accu.stack(), axis=0))
    #         loss_dones = loss_dones.write(step, tf.math.reduce_mean(loss_dones_accu.stack(), axis=0))


    #         action = [None]*self.action_spec_len
    #         for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])

    #         inputs_step_store = {'obs':inputs_step['obs'], 'actions':action, 'step_size':self.step_size_one}
    #         trans_logits = self.trans(inputs_step_store); trans_dist = self.trans.dist(trans_logits)
    #         inputs_step_store['obs'] = trans_dist.sample()
    #         rwd_logits = self.rwd(inputs_step_store); done_logits = self.done(inputs_step_store)


    #         # ## _img
    #         # return_step = inputs['returns'][-1][step:step+1]
    #         # if gen == 0:
    #         #     with tf.GradientTape() as tape_PG:
    #         #         action_logits = self.action(inputs_step, use_img=True, store_real=True)
    #         #         action_dist = [None]*self.action_spec_len
    #         #         for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
    #         #         loss_action = util.loss_PG(action_dist, action, return_step)
    #         #     gradients = tape_PG.gradient(loss_action, self.action.trainable_variables)
    #         #     self.action.optimizer['action'].apply_gradients(zip(gradients, self.action.trainable_variables))
    #         #     loss_PG = loss_PG.write(step, loss_action)

    #         # inputs_act = {'obs':inputs_step['obs'], 'return_goal':return_step, 'step_size':self.step_size_one}
    #         # with tf.GradientTape() as tape_act:
    #         #     action_logits = self.act(inputs_act, use_img=True, store_real=True)
    #         #     action_dist = [None]*self.action_spec_len
    #         #     for i in range(self.action_spec_len): action_dist[i] = self.act.dist[i](action_logits[i])
    #         #     loss_action = util.loss_likelihood(action_dist, action)
    #         # gradients = tape_act.gradient(loss_action, self.act.trainable_variables) # self.rep.trainable_variables +
    #         # self.act.optimizer['act'].apply_gradients(zip(gradients, self.act.trainable_variables)) # self.rep.trainable_variables +
    #         # loss_act = loss_act.write(step, loss_action)


    #     loss['loss_PG_img'], loss['loss_act_img'] = loss_PG_img.concat(), loss_act_img.concat()
    #     loss['trans_ret'], loss['reward_ret'], loss['done_ret'] = loss_trans_ret.concat(), loss_rewards_ret.concat(), loss_dones_ret.concat()
    #     loss['trans'], loss['reward'], loss['done'] = loss_trans.concat(), loss_rewards.concat(), loss_dones.concat()
    #     loss['loss_PG'], loss['loss_act'] = loss_PG.concat(), loss_act.concat()
    #     return loss

    # def MU4_dyn_act_learner(self, inputs, gen, training=True):
    #     print("tracing -> GeneralAI MU4_dyn_act_learner")
    #     loss = {}
    #     # loss_PG_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     # loss_act_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_rewards_ret = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_dones_ret = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_rewards = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_dones = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_PG = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_act = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #     episode_len = tf.shape(inputs['dones'])[0]
    #     for step in tf.range(episode_len):
    #         obs = [None]*self.obs_spec_len
    #         for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])

    #         inputs_step = {'obs':obs, 'actions':self.action_zero_out, 'step_size':self.step_size_one}
    #         with tf.GradientTape(persistent=True) as tape_act:
    #             rep_logits = self.rep(inputs_step, step=step); rep_dist = self.rep.dist(rep_logits)
    #             inputs_step['obs'] = rep_dist.sample()



    #         # loss_PG_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         # loss_act_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_rewards_ret_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_dones_ret_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_rewards_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_dones_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #         self.action.reset_states(use_img=True); self.act.reset_states(use_img=True)
    #         self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
    #         inputs_step_img = {'obs':inputs_step['obs'], 'actions':self.action_zero_out, 'step_size':self.step_size_one}
    #         for step_img in tf.range(step, episode_len):
    #             action = [None]*self.action_spec_len
    #             for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step_img:step_img+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
    #             return_step = inputs['returns'][-1][step_img:step_img+1]

    #             # if gen == 0:
    #             #     with tf.GradientTape() as tape_PG:
    #             #         action_logits = self.action(inputs_step_img, use_img=True)
    #             #         action_dist = [None]*self.action_spec_len
    #             #         for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
    #             #         loss_action = util.loss_PG(action_dist, action, return_step)
    #             #     gradients = tape_PG.gradient(loss_action, self.action.trainable_variables)
    #             #     self.action.optimizer['action'].apply_gradients(zip(gradients, self.action.trainable_variables))
    #             #     # loss_PG_img_accu = loss_PG_img_accu.write(step_img, loss_action)

    #             # inputs_act = {'obs':inputs_step_img['obs'], 'return_goal':return_step, 'step_size':self.step_size_one}
    #             # with tf.GradientTape() as tape_act_img:
    #             #     action_logits = self.act(inputs_act, use_img=True)
    #             #     action_dist = [None]*self.action_spec_len
    #             #     for i in range(self.action_spec_len): action_dist[i] = self.act.dist[i](action_logits[i])
    #             #     loss_action = util.loss_likelihood(action_dist, action)
    #             # gradients = tape_act_img.gradient(loss_action, self.act.trainable_variables)
    #             # self.act.optimizer['act'].apply_gradients(zip(gradients, self.act.trainable_variables))
    #             # # loss_act_img_accu = loss_act_img_accu.write(step_img, loss_action)


    #             step_scale = self.attn_img_scales-1
    #             step_size = tf.math.pow(self.attn_mem_base, step_scale)
    #             inputs_step_img_ret = {'obs':inputs_step_img['obs'], 'actions':action, 'step_size':step_size}
    #             with tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done:
    #                 trans_logits = self.trans(inputs_step_img_ret, store_memory=False, use_img=True); trans_dist = self.trans.dist(trans_logits)
    #                 inputs_step_img_ret['obs'] = trans_dist.sample()

    #             with tape_reward:
    #                 rwd_logits = self.rwd(inputs_step_img_ret, store_memory=False, use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #                 loss_reward = util.loss_likelihood(rwd_dist, inputs['returns'][step_scale][step_img:step_img+1])
    #             gradients = tape_reward.gradient(loss_reward, self.trans.trainable_variables) # + self.rwd.trainable_variables
    #             self.rwd.optimizer['rwd'].apply_gradients(zip(gradients, self.trans.trainable_variables)) # + self.rwd.trainable_variables
    #             loss_rewards_ret_accu = loss_rewards_ret_accu.write(step_img, loss_reward)

    #             with tape_done:
    #                 done_logits = self.done(inputs_step_img_ret, store_memory=False, use_img=True); done_dist = self.done.dist[0](done_logits[0])
    #                 loss_done = util.loss_likelihood(done_dist, tf.constant([[True]]))
    #             gradients = tape_done.gradient(loss_done, self.trans.trainable_variables) # + self.done.trainable_variables
    #             self.done.optimizer['done'].apply_gradients(zip(gradients, self.trans.trainable_variables)) # + self.done.trainable_variables
    #             loss_dones_ret_accu = loss_dones_ret_accu.write(step_img, loss_done)


    #             inputs_step_img['actions'] = action
    #             with tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done:
    #                 trans_logits = self.trans(inputs_step_img, use_img=True); trans_dist = self.trans.dist(trans_logits)
    #                 inputs_step_img['obs'] = trans_dist.sample()

    #             with tape_reward:
    #                 rwd_logits = self.rwd(inputs_step_img, use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #                 loss_reward = util.loss_likelihood(rwd_dist, inputs['rewards'][step_img:step_img+1])
    #             gradients = tape_reward.gradient(loss_reward, self.trans.trainable_variables) # + self.rwd.trainable_variables
    #             self.rwd.optimizer['rwd'].apply_gradients(zip(gradients, self.trans.trainable_variables)) # + self.rwd.trainable_variables
    #             loss_rewards_accu = loss_rewards_accu.write(step_img, loss_reward)

    #             with tape_done:
    #                 done_logits = self.done(inputs_step_img, use_img=True); done_dist = self.done.dist[0](done_logits[0])
    #                 loss_done = util.loss_likelihood(done_dist, inputs['dones'][step_img:step_img+1])
    #             gradients = tape_done.gradient(loss_done, self.trans.trainable_variables) # + self.done.trainable_variables
    #             self.done.optimizer['done'].apply_gradients(zip(gradients, self.trans.trainable_variables)) # + self.done.trainable_variables
    #             loss_dones_accu = loss_dones_accu.write(step_img, loss_done)

    #         # loss_PG_img = loss_PG_img.write(step, tf.math.reduce_mean(loss_PG_img_accu.stack(), axis=0))
    #         # loss_act_img = loss_act_img.write(step, tf.math.reduce_mean(loss_act_img_accu.stack(), axis=0))
    #         loss_rewards_ret = loss_rewards_ret.write(step, tf.math.reduce_mean(loss_rewards_ret_accu.stack(), axis=0))
    #         loss_dones_ret = loss_dones_ret.write(step, tf.math.reduce_mean(loss_dones_ret_accu.stack(), axis=0))
    #         loss_rewards = loss_rewards.write(step, tf.math.reduce_mean(loss_rewards_accu.stack(), axis=0))
    #         loss_dones = loss_dones.write(step, tf.math.reduce_mean(loss_dones_accu.stack(), axis=0))



    #         action = [None]*self.action_spec_len
    #         for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
    #         return_step = inputs['returns'][-1][step:step+1]

    #         step_scale = self.attn_img_scales-1
    #         step_size = tf.math.pow(self.attn_mem_base, step_scale)
    #         inputs_step_store = {'obs':inputs_step['obs'], 'actions':action, 'step_size':step_size}
    #         trans_logits = self.trans(inputs_step_store, store_memory=False); trans_dist = self.trans.dist(trans_logits)
    #         inputs_step_store['obs'] = trans_dist.sample()
    #         rwd_logits = self.rwd(inputs_step_store, store_memory=False); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #         loss_return = util.loss_likelihood(rwd_dist, inputs['returns'][step_scale][step:step+1])

    #         inputs_step_store = {'obs':inputs_step['obs'], 'actions':action, 'step_size':self.step_size_one}
    #         trans_logits = self.trans(inputs_step_store); trans_dist = self.trans.dist(trans_logits)
    #         inputs_step_store['obs'] = trans_dist.sample()
    #         rwd_logits = self.rwd(inputs_step_store); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #         loss_reward = util.loss_likelihood(rwd_dist, inputs['rewards'][step:step+1])
    #         done_logits = self.done(inputs_step_store); done_dist = self.done.dist[0](done_logits[0])
    #         loss_done = util.loss_likelihood(done_dist, inputs['dones'][step:step+1])


    #         if gen == 0:
    #             with tf.GradientTape() as tape_PG:
    #                 action_logits = self.action(inputs_step, use_img=True, store_real=True)
    #                 action_dist = [None]*self.action_spec_len
    #                 for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
    #                 loss_action = util.loss_PG(action_dist, action, return_step)
    #                 # surprise = -loss_return-loss_reward-loss_done
    #                 # surprise = loss_return+loss_reward+loss_done
    #                 # loss_action = util.loss_PG(action_dist, action, surprise)
    #             gradients = tape_PG.gradient(loss_action, self.action.trainable_variables)
    #             self.action.optimizer['action'].apply_gradients(zip(gradients, self.action.trainable_variables))
    #             loss_PG = loss_PG.write(step, loss_action)

    #         # inputs_act = {'obs':inputs_step['obs'], 'return_goal':return_step, 'step_size':self.step_size_one}
    #         # with tape_act:
    #         #     action_logits = self.act(inputs_act, use_img=True, store_real=True)
    #         #     action_dist = [None]*self.action_spec_len
    #         #     for i in range(self.action_spec_len): action_dist[i] = self.act.dist[i](action_logits[i])
    #         #     loss_action = util.loss_likelihood(action_dist, action)
    #         #     # surprise = -loss_return-loss_reward-loss_done
    #         #     # surprise = loss_return+loss_reward+loss_done
    #         #     # loss_action = util.loss_PG(action_dist, action, surprise)
    #         # gradients = tape_act.gradient(loss_action, self.rep.trainable_variables + self.act.trainable_variables)
    #         # self.act.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.act.trainable_variables))
    #         # loss_act = loss_act.write(step, loss_action)


    #     # loss['loss_PG_img'], loss['loss_act_img'] = loss_PG_img.concat(), loss_act_img.concat()
    #     loss['reward_ret'], loss['done_ret'] = loss_rewards_ret.concat(), loss_dones_ret.concat()
    #     loss['reward'], loss['done'] = loss_rewards.concat(), loss_dones.concat()
    #     loss['loss_PG'], loss['loss_act'] = loss_PG.concat(), loss_act.concat()
    #     return loss

    # def MU4_dyn_learner(self, inputs, training=True):
    #     print("tracing -> GeneralAI MU4_dyn_learner")
    #     loss = {}
    #     loss_rewards = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_dones = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #     episode_len = tf.shape(inputs['dones'])[0]
    #     for step in tf.range(episode_len-2):
    #         action = [None]*self.action_spec_len
    #         for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
    #         inputs_step = {'obs':inputs['obs_rep'][step:step+1][0], 'actions':action}
    #         self.trans(inputs_step)
    #         inputs_step['obs'] = inputs['obs_trans'][step:step+1][0]
    #         self.rwd(inputs_step); self.done(inputs_step)
    #         self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)

    #         action = [None]*self.action_spec_len
    #         for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step+1:step+2]; action[i].set_shape(self.action_spec[i]['step_shape'])
    #         inputs_step['actions'] = action
    #         self.trans(inputs_step, use_img=True)
    #         inputs_step['obs'] = inputs['obs_trans_img'][step+1:step+2][0]
    #         self.rwd(inputs_step, use_img=True); self.done(inputs_step, use_img=True)

    #         loss_rewards_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_dones_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         for step_img in tf.range(step+2, episode_len):
    #             action = [None]*self.action_spec_len
    #             for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step_img:step_img+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
    #             inputs_step['actions'] = action

    #             with tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done:
    #                 trans_logits = self.trans(inputs_step, use_img=True); trans_dist = self.trans.dist(trans_logits)
    #                 inputs_step['obs'] = trans_dist.sample()

    #             with tape_reward:
    #                 rwd_logits = self.rwd(inputs_step, use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #                 loss_reward = util.loss_likelihood(rwd_dist, inputs['rewards'][step_img:step_img+1])
    #             gradients = tape_reward.gradient(loss_reward, self.trans.trainable_variables + self.rwd.trainable_variables)
    #             self.rwd.optimizer['rwd'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.rwd.trainable_variables))
    #             loss_rewards_img_accu = loss_rewards_img_accu.write(step_img, loss_reward)

    #             with tape_done:
    #                 done_logits = self.done(inputs_step, use_img=True); done_dist = self.done.dist[0](done_logits[0])
    #                 loss_done = util.loss_likelihood(done_dist, inputs['dones'][step_img:step_img+1])
    #             gradients = tape_done.gradient(loss_done, self.trans.trainable_variables + self.done.trainable_variables)
    #             self.done.optimizer['done'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.done.trainable_variables))
    #             loss_dones_img_accu = loss_dones_img_accu.write(step_img, loss_done)

    #         loss_rewards = loss_rewards.write(step, tf.math.reduce_mean(loss_rewards_img_accu.stack(), axis=0))
    #         loss_dones = loss_dones.write(step, tf.math.reduce_mean(loss_dones_img_accu.stack(), axis=0))

    #     loss['reward'], loss['done'] = loss_rewards.concat(), loss_dones.concat()
    #     return loss

    def MU4(self):
        print("tracing -> GeneralAI MU4")
        num_gen = 2
        ma, snr, std, loss_meta = tf.constant(0,tf.float64), tf.constant(1,tf.float64), tf.constant(0,tf.float64), tf.constant([0],self.compute_dtype)
        ma_loss, snr_loss, std_loss = tf.constant(0,self.compute_dtype), tf.constant(1,self.compute_dtype), tf.constant(0,self.compute_dtype)
        ma_loss_act, snr_loss_act, std_loss_act = tf.constant(0,self.compute_dtype), tf.constant(1,self.compute_dtype), tf.constant(0,self.compute_dtype)
        episode, stop = tf.constant(0), tf.constant(False)
        while episode < self.max_episodes*num_gen and not stop:
            tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}

            gen, episode_gen = episode%num_gen, episode//num_gen
            log_metrics, train = [True,True,True,True]+[True,True,True]+[True,True]+[True,True]+[True,True,True], True
            return_goal = tf.constant([[200.0]], tf.float64)
            # return_goal = tf.reshape((ma + 10.0),(1,1)) # _rpP
            return_goal_alt = tf.constant([[200.0]], tf.float64)
            # return_goal_alt = tf.random.uniform((1,1), minval=0.0, maxval=200.0, dtype=tf.float64)
            if gen == 0: return_goal, log_metrics, train, gen = return_goal, [False,False,False,False]+[True,True,True]+[False,False]+[True,True]+[True,True,True], True, 0 # action/PG
            if gen == 1: return_goal, log_metrics, train, gen = return_goal, [True,True,True,True]+[False,False,False]+[True,True]+[False,False]+[False,False,False], True, 1 # actout/act # TODO if was not relabling, training here would be like epocs
            if gen == 2: return_goal, log_metrics, train, gen = return_goal, [False,False,False,False]+[False,False,False]+[False,False]+[False,False]+[False,False,False], True, 2 # random, actionL/PGL
            if gen == 3: return_goal, log_metrics, train, gen = return_goal_alt, [False,False,False,False]+[False,False,False]+[False,False]+[False,False]+[False,False,False], True, 1 # act alt

            self.reset_states(); outputs, inputs, loss_actor = self.MU4_actor(inputs, gen, return_goal, return_goal_alt)
            rewards_total = outputs['returns'][-1][0][0] # tf.math.reduce_mean(outputs['rewards'])
            if gen == 0:
                util.stats_update(self.action.stats['rwd'], rewards_total); ma, _, snr, std = util.stats_get(self.action.stats['rwd'])

                # _, _, _, std_loss = util.stats_get(self.action.stats['loss'])
                # obs = [self.action.stats['loss']['iter'].value(), tf.cast(ma,self.compute_dtype), std_loss]
                # inputs_meta = {'obs':[tf.expand_dims(tf.stack(obs,0),0)]}

                # learn_rate = self.action.optimizer['action'].learning_rate
                # with tf.GradientTape() as tape_meta:
                #     meta_logits = self.meta(inputs_meta); meta_dist = self.meta.dist[0](meta_logits[0])
                #     loss_meta = util.loss_PG(meta_dist, tf.reshape(learn_rate,(1,1)), tf.reshape(rewards_total,(1,1)))
                # gradients = tape_meta.gradient(loss_meta, self.meta.trainable_variables)
                # self.meta.optimizer['meta'].apply_gradients(zip(gradients, self.meta.trainable_variables))

                # meta_logits = self.meta(inputs_meta); meta_dist = self.meta.dist[0](meta_logits[0])
                # learn_rate = meta_dist.sample()
                # self.action.optimizer['action'].learning_rate = util.discretize(learn_rate, self.meta_spec[0])
                # # self.action.optimizer['action'].learning_rate = tf.squeeze(tf.cast(learn_rate, tf.float64))

                # self.action.optimizer['action'].learning_rate = self.action_get_learn_rate(ma_loss) # _lr-loss
                # self.action.optimizer['action'].learning_rate = self.action_get_learn_rate(std) # _lr-rwd-std
                # self.action.optimizer['action'].learning_rate = tf.math.exp(episode / self.max_episodes * (-16.0 + 11.0) - 11.0) # _lr-scale
                # self.action.optimizer['action'].learning_rate = self.learn_rate * snr_loss # _lr-loss-snr-scale
                # self.action.optimizer['action'].learning_rate = self.learn_rate * snr_loss**3 # _lr-loss-snr-scale3

            if gen == 1:
                util.stats_update(self.act.stats['rwd'], rewards_total); ma, _, _, _ = util.stats_get(self.act.stats['rwd'])
                self.act.optimizer['act'].learning_rate = self.act_get_learn_rate(tf.math.abs(return_goal[0][0] - ma))
            if gen == 2: util.stats_update(self.actionL.stats['rwd'], rewards_total); ma, _, _, _ = util.stats_get(self.actionL.stats['rwd'])


            loss_rep = {'act':tf.constant([0],self.compute_dtype), 'act':tf.constant([0],self.compute_dtype)}
            loss_act = {'act':tf.constant([0],self.compute_dtype), 'PG':tf.constant([0],self.compute_dtype)}
            loss_PG = {'PG':tf.constant([0],self.compute_dtype)}
            loss_dyn = {'trans':tf.constant([0],self.compute_dtype), 'reward':tf.constant([0],self.compute_dtype), 'done':tf.constant([0],self.compute_dtype)}
            if train:
                self.reset_states(); loss_rep = self.MU4_rep_learner(outputs, gen) # _repL1
                self.reset_states(); outputs_rep = self.MU4_rep_returns(outputs)
                # self.reset_states(); loss_act = self.MU4_act_learner(outputs, return_goal)
                # if gen == 0: self.reset_states(); loss_PG = self.MU4_PG_learner(outputs)
                # self.reset_states(); loss_dyn = self.MU4_dyn_learner(outputs) # _dyn4
                # self.reset_states(); loss_dyn = self.MU4_dyn_act_learner(outputs, gen) # _dyn5
                # self.reset_states(); loss_dyn = self.MU4_dyn_learner2(outputs, gen) # _dyn8
                # self.reset_states(); loss_dyn = self.MU4_dyn_learner3(outputs) # _dyn9
                # self.reset_states(); loss_dyn = self.MU4_dyn_learner4(outputs, outputs_rep) # _dynA
                self.reset_states(); loss_act = self.MU4_act_PG_learner(outputs, outputs_rep, gen, return_goal)

                if gen == 0:
                    util.stats_update(self.action.stats['loss'], tf.math.reduce_mean(loss_act['PG'])); ma_loss, _, snr_loss, std_loss = util.stats_get(self.action.stats['loss'])
                    # util.stats_update(self.act.stats['loss'], tf.math.reduce_mean(loss_act['act'])); ma_loss_act, _, snr_loss_act, std_loss_act = util.stats_get(self.act.stats['loss'])
                    # maL, _, _, _ = util.stats_get(self.actionL.stats['rwd'])
                    # if self.action.stats['loss']['iter'] > 10 and std_loss < 1.0 and tf.math.abs(ma_loss) < 1.0:
                    # if snr_loss < 0.5 and std_loss < 0.1 and tf.math.abs(ma_loss) < 0.1:
                    # if snr_loss < 0.5 and tf.math.abs(ma_loss) < 0.1:
                    if self.action.stats['loss']['iter'] > 10 and tf.math.abs(ma_loss) < 0.05:
                        # if ma > maL: util.net_copy(self.action, self.actionL)
                        util.net_reset(self.action) #; util.net_reset(self.rep)
                        # self.action.optimizer['action'].learning_rate = tf.random.uniform((), dtype=tf.float64, maxval=self.learn_rate, minval=self.float64_eps) # _lr-rnd-linear
                        # self.action.optimizer['action'].learning_rate = tf.math.exp(tf.random.uniform((), dtype=tf.float64, maxval=-7, minval=-16)) # _lr-rnd-exp
                        tf.print("net_reset (action) at:", episode_gen, " lr:", self.action.optimizer['action'].learning_rate, " ma_loss:", ma_loss, " snr_loss:", snr_loss, " std_loss:", std_loss) # , " copy?", ma > maL

                if gen == 1: util.stats_update(self.act.stats['loss'], tf.math.reduce_mean(loss_act['act'])); ma_loss, _, snr_loss, std_loss = util.stats_get(self.act.stats['loss'])
                if gen == 2: util.stats_update(self.actionL.stats['loss'], tf.math.reduce_mean(loss_act['PG'])); ma_loss, _, snr_loss, std_loss = util.stats_get(self.actionL.stats['loss'])


            metrics = [log_metrics, episode_gen, ma, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                ma, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0],
                # ma, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0],
                # tf.math.reduce_mean(loss_actor['returns_pred']),
                ma_loss, tf.math.reduce_mean(loss_act['act']),
                ma_loss, tf.math.reduce_mean(loss_act['PG']),
                # snr_loss, std_loss,
                # tf.math.reduce_mean(loss_rep['trans']),
                # tf.math.reduce_mean(loss_rep['reward']), tf.math.reduce_mean(loss_rep['done']),
                # tf.math.reduce_mean(loss_actor['trans']),
                # tf.math.reduce_mean(loss_actor['reward']), tf.math.reduce_mean(loss_actor['done']),
                # tf.math.reduce_mean(loss_actor['trans_img']),
                # tf.math.reduce_mean(loss_actor['reward_img']), tf.math.reduce_mean(loss_actor['done_img']),
                # tf.math.reduce_mean(loss_dyn['trans']),
                # tf.math.reduce_mean(loss_dyn['reward']), tf.math.reduce_mean(loss_dyn['done']),
                # tf.math.reduce_mean(loss_actor['action']),
                # tf.math.reduce_mean(loss_actor['entropy']),
                # self.action.optimizer['action'].learning_rate,
                # loss_meta[0],
            ]
            if self.trader:
                del metrics[2]; metrics[2], metrics[3] = tf.math.reduce_mean(tf.concat([outputs['obs'][3],inputs['obs'][3]],0)), inputs['obs'][3][-1][0]
                metrics += [tf.math.reduce_mean(tf.concat([outputs['obs'][4],inputs['obs'][4]],0)), tf.math.reduce_mean(tf.concat([outputs['obs'][5],inputs['obs'][5]],0)), inputs['obs'][0][-1][0] - outputs['obs'][0][0][0],]
            dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

            stop = tf.numpy_function(self.check_stop, [tf.constant(0)], tf.bool); stop.set_shape(())
            episode += 1




def params(): pass
load_model, save_model = False, False
max_episodes = 10
learn_rate = 2e-5 # 5 = testing, 6 = more stable/slower # tf.experimental.numpy.finfo(tf.float64).eps
value_cont = True
latent_size = 16
latent_dist = 'd' # 'd' = deterministic, 'c' = categorical, 'mx' = continuous(mix-log)
mixture_multi = 4
net_attn_io = True
aio_max_latents = 64
attn_mem_base = 4 # max_steps must be power of this!
aug_data_step, aug_data_pos = True, False

device_type = 'GPU' # use GPU for large networks (over 8 total net blocks?) or output data (512 bytes?)
device_type = 'CPU'

machine, device, extra = 'dev', 1, '_lr2e5_rep2e8_gen01-pg6e6-rp200' # _repL1 _gen0123-rnd-pg2e5 _dyn1279 _rp200-rnd _img _prs2 _wd7 _train _RfB _entropy3 _mae _perO-NR-NT-G-Nrez _rez-rezoR-rezoT-rezoG _mixlog-abs-log1p-Nreparam _obs-tsBoxF-dataBoxI_round _Nexp-Ne9-Nefmp36-Nefmer154-Nefme308-emr-Ndiv _MUimg-entropy-values-policy-Netoe _AC-Nonestep-aing _stepE _cncat

trader, env_async, env_async_clock, env_async_speed, env_reconfig = False, False, 0.001, 160.0, False
env_name, max_steps, env_render, env_reconfig, env = 'CartPole', 256, False, True, gym.make('CartPole-v0') # ; env.observation_space.dtype = np.dtype('float64') # (4) float32    ()2 int64    200  195.0
# env_name, max_steps, env_render, env_reconfig, env = 'CartPole', 512, False, True, gym.make('CartPole-v1') # ; env.observation_space.dtype = np.dtype('float64') # (4) float32    ()2 int64    500  475.0
# env_name, max_steps, env_render, env_reconfig, env = 'LunarLand', 1024, False, True, gym.make('LunarLander-v2') # (8) float32    ()4 int64    1000  200
# env_name, max_steps, env_render, env = 'Copy', 256, False, gym.make('Copy-v0') # DuplicatedInput-v0 RepeatCopy-v0 Reverse-v0 ReversedAddition-v0 ReversedAddition3-v0 # ()6 int64    [()2,()2,()5] int64    200  25.0
# env_name, max_steps, env_render, env = 'ProcgenChaser', 1024, False, gym.make('procgen-chaser-v0') # (64,64,3) uint8    ()15 int64    1000 None
# env_name, max_steps, env_render, env = 'ProcgenCaveflyer', 1024, False, gym.make('procgen-caveflyer-v0') # (64,64,3) uint8    ()15 int64    1000 None
# env_name, max_steps, env_render, env = 'Tetris', 22528, False, gym.make('ALE/Tetris-v5') # (210,160,3) uint8    ()18 int64    21600 None
# env_name, max_steps, env_render, env = 'MontezumaRevenge', 22528, False, gym.make('MontezumaRevengeNoFrameskip-v4') # (210,160,3) uint8    ()18 int64    400000 None
# env_name, max_steps, env_render, env = 'MsPacman', 22528, False, gym.make('MsPacmanNoFrameskip-v4') # (210,160,3) uint8    ()9 int64    400000 None

# env_name, max_steps, env_render, env_reconfig, env = 'CartPoleCont', 256, False, True, gym.make('CartPoleContinuousBulletEnv-v0'); env.observation_space.dtype = np.dtype('float64') # (4) float32    (1) float32    200  190.0
# env_name, max_steps, env_render, env_reconfig, env = 'LunarLandCont', 1024, False, True, gym.make('LunarLanderContinuous-v2') # (8) float32    (2) float32    1000  200
# import envs_local.bipedal_walker as env_; env_name, max_steps, env_render, env = 'BipedalWalker', 2048, False, env_.BipedalWalker()
# env_name, max_steps, env_render, env = 'Hopper', 1024, False, gym.make('HopperBulletEnv-v0') # (15) float32    (3) float32    1000  2500.0
# env_name, max_steps, env_render, env = 'RacecarZed', 1024, False, gym.make('RacecarZedBulletEnv-v0') # (10,100,4) uint8    (2) float32    1000  5.0

# from pettingzoo.butterfly import pistonball_v4; env_name, max_steps, env_render, env = 'PistonBall', 1, False, pistonball_v4.env()

# import envs_local.random_env as env_; env_name, max_steps, env_render, env = 'TestRnd', 64, False, env_.RandomEnv(True)
# import envs_local.data_env as env_; env_name, max_steps, env_render, env = 'DataShkspr', 64, False, env_.DataEnv('shkspr')
# # import envs_local.data_env as env_; env_name, max_steps, env_render, env = 'DataMnist', 64, False, env_.DataEnv('mnist')
# import gym_trader; tenv = 2; env_name, max_steps, env_render, env, trader = 'Trader'+str(tenv), 1024, False, gym.make('Trader-v0', agent_id=device, env=tenv), True

# max_steps = 64 # max replay buffer or train interval or bootstrap

arch = 'MU4' # Dreamer/planner w/imagination+generalization

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
    if env_reconfig: import envs_local.reconfig_wrapper as envrw_; env_name, env = env_name+'-r', envrw_.ReconfigWrapperEnv(env)
    with tf.device("/device:{}:{}".format(device_type,(device if device_type=='GPU' else 0))):
        model = GeneralAI(arch, env, trader, env_render, max_episodes, max_steps, learn_rate, value_cont, latent_size, latent_dist, mixture_multi, net_attn_io, aio_max_latents, attn_mem_base, aug_data_step, aug_data_pos)
        name = "gym-{}-{}".format(arch, env_name)

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
            model_name = "{}-{}-a{}".format(net.arch_desc, machine, device)
            model_file = "{}/tf-data-models-local/{}.h5".format(curdir, model_name); loaded_model = False
            model_files[net.name] = model_file
            if (load_model or net.name == 'M') and tf.io.gfile.exists(model_file):
                net.load_weights(model_file, by_name=True, skip_mismatch=True)
                print("LOADED {} weights from {}".format(net.name, model_file)); loaded_model = True
            name_opt = "-O{}{}".format(net.opt_spec['type'], ('' if net.opt_spec['schedule_type']=='' else '-S'+net.opt_spec['schedule_type'])) if hasattr(net, 'opt_spec') else ''
            name_arch += "   {}{}-{}".format(net.arch_desc, name_opt, 'load' if loaded_model else 'new')


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
                for j in range(len(loss_group[k])): loss_group[k][j] = 0 if loss_group[k][j] == [] else np.mean(loss_group[k][j])
        # TODO np.mean, reduce size if above 200,000 episodes

        name = "{}-{}-a{}{}-{}".format(name, machine, device, extra, time.strftime("%y-%m-%d-%H-%M-%S"))
        total_steps = int(np.sum(metrics_loss['1steps']['steps+']))
        step_time = total_time/total_steps
        title = "{}    [{}-{}] {}\ntime:{}    steps:{}    t/s:{:.8f}".format(name, device_type, tf.keras.backend.floatx(), name_arch, util.print_time(total_time), total_steps, step_time)
        title += "     |     lr:{:.0e}    al:{}    am:{}    ms:{}".format(learn_rate, aio_max_latents, attn_mem_base, max_steps)
        title += "     |     a-clk:{}    a-spd:{}    aug:{}{}    aio:{}".format(env_async_clock, env_async_speed, ('S' if aug_data_step else ''), ('P' if aug_data_pos else ''), ('Y' if net_attn_io else 'N')); print(title)

        import matplotlib as mpl
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue','lightblue','green','lime','red','lavender','turquoise','cyan','magenta','salmon','yellow','gold','black','brown','purple','pink','orange','teal','coral','darkgreen','tan'])
        plt.figure(num=name, figsize=(34, 18), tight_layout=True)
        xrng, i, vplts, lim = np.arange(0, max_episodes, 1), 0, 0, 0.03
        for loss_group_name in metrics_loss.keys(): vplts += int(loss_group_name[0])

        for loss_group_name, loss_group in metrics_loss.items():
            rows, col, m_min, m_max, combine, yscale = int(loss_group_name[0]), 0, [0]*len(loss_group), [0]*len(loss_group), loss_group_name.endswith('*'), ('log' if loss_group_name[1] == '~' else 'linear')
            if combine: spg = plt.subplot2grid((vplts, 1), (i, 0), rowspan=rows, xlim=(0, max_episodes), yscale=yscale); plt.grid(axis='y',alpha=0.3)
            for metric_name, metric in loss_group.items():
                metric = np.asarray(metric, np.float64); m_min[col], m_max[col] = np.nanquantile(metric, lim), np.nanquantile(metric, 1.0-lim)
                if not combine: spg = plt.subplot2grid((vplts, len(loss_group)), (i, col), rowspan=rows, xlim=(0, max_episodes), ylim=(m_min[col], m_max[col]), yscale=yscale); plt.grid(axis='y',alpha=0.3)
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
