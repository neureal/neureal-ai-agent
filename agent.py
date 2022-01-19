from collections import OrderedDict
import time, os # , talib, bottleneck
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
# tf.random.set_seed(0)
tf.keras.backend.set_epsilon(tf.experimental.numpy.finfo(tf.keras.backend.floatx()).eps) # 1e-7 default
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import model_util as util
import gym, gym_algorithmic, procgen, pybullet_envs

# CUDA 11.2.2_461.33, CUDNN 8.1.1.33, tensorflow-gpu==2.6.0, tensorflow_probability==0.14.0
physical_devices_gpu = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices_gpu)): tf.config.experimental.set_memory_growth(physical_devices_gpu[i], True)

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


class RepNet(tf.keras.Model):
    def __init__(self, name, spec_in, latent_spec, latent_dist, latent_size, net_blocks=0, net_attn=False, net_lstm=False, net_attn_io=False, net_attn_io2=False, num_heads=1, memory_size=None, max_steps=1, aug_data_step=False, aug_data_pos=False):
        super(RepNet, self).__init__(name=name)
        inp, mid, evo = latent_size*4, latent_size*2, int(latent_size/2)
        self.net_blocks, self.net_attn, self.net_lstm, self.net_attn_io2, self.aug_data_step = net_blocks, net_attn, net_lstm, net_attn_io2, aug_data_step
        self.layer_flatten = tf.keras.layers.Flatten()

        # TODO how to loop through inputs?
        # self.net_inputs = ['obs']*len(spec_in)+['rewards','dones']
        self.net_ins = len(spec_in); self.net_ins_all, self.layer_attn_in, self.layer_mlp_in, self.pos_idx_in = self.net_ins, [None]*self.net_ins, [None]*self.net_ins, [None]*self.net_ins
        for i in range(self.net_ins):
            event_shape, event_size, channels, num_latents = spec_in[i]['event_shape'], spec_in[i]['event_size'], spec_in[i]['channels'], spec_in[i]['num_latents']
            if aug_data_pos and event_size > 1:
                    pos_idx = np.indices(spec_in[i]['event_shape'][:-1])
                    pos_idx = np.moveaxis(pos_idx, 0, -1)
                    # pos_idx = pos_idx / (np.max(pos_idx).item() / 2.0) - 1.0
                    self.pos_idx_in[i] = tf.constant(pos_idx, dtype=self.compute_dtype)
                    channels += pos_idx.shape[-1]
            if net_attn_io and event_size > 1:
                self.layer_attn_in[i] = util.MultiHeadAttention(latent_size=latent_size, num_heads=1, norm=True, hidden_size=inp, evo=evo, residual=False, cross_type=1, num_latents=num_latents, channels=channels, name='attn_in_{:02d}'.format(i))
            self.layer_mlp_in[i] = util.MLPBlock(hidden_size=inp, latent_size=latent_size, evo=evo, residual=False, name='mlp_in_{:02d}'.format(i))
        if aug_data_step:
            self.net_ins_all += 1
            # self.layer_rwd_in = util.MLPBlock(hidden_size=inp, latent_size=latent_size, evo=evo, residual=False, name='rwd_in')
            # self.layer_done_in = util.MLPBlock(hidden_size=inp, latent_size=latent_size, evo=evo, residual=False, name='done_in')
            self.layer_step_in = util.MLPBlock(hidden_size=inp, latent_size=latent_size, evo=evo, residual=False, name='step_in')
        if net_attn_io2: self.layer_attn_io2 = util.MultiHeadAttention(latent_size=latent_size, num_heads=1, norm=False, residual=False, cross_type=1, num_latents=latent_spec['num_latents'], channels=latent_size, name='attn_io2')

        self.layer_attn, self.layer_lstm, self.layer_mlp = [], [], []
        for i in range(net_blocks):
            if self.net_attn:
                self.layer_attn += [util.MultiHeadAttention(latent_size=latent_size, num_heads=num_heads, memory_size=memory_size, residual=True, name='attn_{:02d}'.format(i))]
                self.layer_mlp += [util.MLPBlock(hidden_size=mid, latent_size=latent_size, evo=None, residual=True, name='mlp_{:02d}'.format(i))]
            elif self.net_lstm:
                self.layer_lstm += [tf.keras.layers.LSTM(mid, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i))]
                self.layer_mlp += [tf.keras.layers.Dense(latent_size, name='dense_{:02d}'.format(i))]
            else: self.layer_mlp += [util.MLPBlock(hidden_size=mid, latent_size=latent_size, evo=evo, residual=False, name='mlp_{:02d}'.format(i))]

        if latent_dist == 0: params_size, self.dist = util.Deterministic.params_size(latent_spec['event_shape']), util.Deterministic(latent_spec['event_shape'])
        if latent_dist == 1: params_size, self.dist = util.CategoricalRP.params_size(latent_spec['event_shape']), util.CategoricalRP(latent_spec['event_shape'])
        if latent_dist == 2: params_size, self.dist = util.MixtureLogistic.params_size(latent_spec['num_components'], latent_spec['event_shape']), util.MixtureLogistic(latent_spec['num_components'], latent_spec['event_shape'])
        self.layer_dense_out_logits = tf.keras.layers.Dense(params_size, name='dense_out_logits')

        self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        self.net_arch = "{}[inD{}-{:02d}{}{}D{}{}-lat{}x{}-{}]".format(name, inp, net_blocks, ('AT+' if self.net_attn else ''), ('LS+' if self.net_lstm else ''), mid, ('-hds'+str(num_heads) if self.net_attn else ''), latent_spec['num_latents'], latent_size, latent_spec['num_components'])

    def reset_states(self, use_img=False):
        for layer in self.layer_attn: layer.reset_states(use_img=use_img)
        for layer in self.layer_lstm: layer.reset_states()
    def call(self, inputs, store_memory=True, use_img=False, store_real=False, step=None, training=None):
        out_accu = [None]*self.net_ins_all
        for i in range(self.net_ins):
            out = tf.cast(inputs['obs'][i], self.compute_dtype)
            # out = tf.expand_dims(out, axis=-1) # TODO try splitting down to individual scaler level
            if self.pos_idx_in[i] is not None:
                shape = tf.concat([tf.shape(out)[0:1], self.pos_idx_in[i].shape], axis=0)
                pos_idx = tf.broadcast_to(self.pos_idx_in[i], shape)
                out = tf.concat([out, pos_idx], axis=-1)
            if self.layer_attn_in[i] is not None: out = self.layer_attn_in[i](out)
            else: out = self.layer_flatten(out)
            out_accu[i] = self.layer_mlp_in[i](out)
        if self.aug_data_step:
            # out = tf.cast(inputs['rewards'], self.compute_dtype)
            # out_accu[-3] = self.layer_rwd_in(out)
            # out = tf.cast(inputs['dones'], self.compute_dtype)
            # out_accu[-2] = self.layer_done_in(out)
            step = tf.reshape(tf.cast(step, self.compute_dtype), [1,1])
            out_accu[-1] = self.layer_step_in(step)
        # out = tf.math.add_n(out_accu) # out = tf.math.accumulate_n(out_accu)
        out = tf.concat(out_accu, axis=0)
        if self.net_attn_io2: out = self.layer_attn_io2(out)
        
        for i in range(self.net_blocks):
            if self.net_attn: out = tf.squeeze(self.layer_attn[i](tf.expand_dims(out, axis=0), auto_mask=training, store_memory=store_memory, use_img=use_img, store_real=store_real), axis=0)
            if self.net_lstm: out = tf.squeeze(self.layer_lstm[i](tf.expand_dims(out, axis=0), training=training), axis=0)
            out = self.layer_mlp[i](out)

        out = self.layer_dense_out_logits(out)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print('rep net out:', out)
        return out


# transition dynamics within latent space
class TransNet(tf.keras.Model):
    def __init__(self, name, spec_in, latent_spec, latent_dist, latent_size, net_blocks=0, net_attn=False, net_lstm=False, net_attn_io=False, num_heads=1, memory_size=None, max_steps=1): # spec_in=[] for no action conditioning
        super(TransNet, self).__init__(name=name)
        inp, mid, evo = latent_size*4, latent_size*2, int(latent_size/2)
        self.net_blocks, self.net_attn, self.net_lstm, self.net_attn_io, self.lat_batch_size = net_blocks, net_attn, net_lstm, net_attn_io, latent_spec['num_latents']
        self.layer_flatten = tf.keras.layers.Flatten()

        # self.net_inputs = ['actions']*len(spec_in)+['obs'] # action conditioning/embedding
        self.net_ins = len(spec_in); self.net_ins_all, self.layer_attn_in, self.layer_mlp_in = self.net_ins+2, [None]*self.net_ins, [None]*self.net_ins
        for i in range(self.net_ins):
            event_shape, event_size, channels, num_latents = spec_in[i]['event_shape'], spec_in[i]['event_size'], spec_in[i]['channels'], spec_in[i]['num_latents']
            # TODO add aug_data_pos?
            if net_attn_io and event_size > 1:
                self.layer_attn_in[i] = util.MultiHeadAttention(latent_size=latent_size, num_heads=1, norm=True, hidden_size=inp, evo=evo, residual=False, cross_type=1, num_latents=num_latents, channels=channels, name='attn_in_{:02d}'.format(i))
            self.layer_mlp_in[i] = util.MLPBlock(hidden_size=inp, latent_size=latent_size, evo=evo, residual=False, name='mlp_in_{:02d}'.format(i))
        self.layer_step_size_in = util.MLPBlock(hidden_size=inp, latent_size=latent_size, evo=evo, residual=False, name='step_size_in')
        # TODO add net_attn_io2
        # TODO add aug_data_step

        self.layer_attn, self.layer_lstm, self.layer_mlp = [], [], []
        for i in range(net_blocks):
            if self.net_attn:
                self.layer_attn += [util.MultiHeadAttention(latent_size=latent_size, num_heads=num_heads, memory_size=memory_size, residual=True, name='attn_{:02d}'.format(i))]
                self.layer_mlp += [util.MLPBlock(hidden_size=mid, latent_size=latent_size, evo=None, residual=True, name='mlp_{:02d}'.format(i))]
            elif self.net_lstm:
                self.layer_lstm += [tf.keras.layers.LSTM(mid, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i))]
                self.layer_mlp += [tf.keras.layers.Dense(latent_size, name='dense_{:02d}'.format(i))]
            else: self.layer_mlp += [util.MLPBlock(hidden_size=mid, latent_size=latent_size, evo=evo, residual=False, name='mlp_{:02d}'.format(i))]
        
        if latent_dist == 0: params_size, self.dist = util.Deterministic.params_size(latent_spec['event_shape']), util.Deterministic(latent_spec['event_shape'])
        if latent_dist == 1: params_size, self.dist = util.CategoricalRP.params_size(latent_spec['event_shape']), util.CategoricalRP(latent_spec['event_shape'])
        if latent_dist == 2: params_size, self.dist = util.MixtureLogistic.params_size(latent_spec['num_components'], latent_spec['event_shape']), util.MixtureLogistic(latent_spec['num_components'], latent_spec['event_shape'])
        # if net_attn_io: self.layer_attn_out = util.MultiHeadAttention(latent_size=params_size, num_heads=1, norm=True, hidden_size=mid, evo=evo, residual=False, cross_type=1, num_latents=self.lat_batch_size, channels=latent_size, name='attn_out')
        if net_attn_io: self.layer_attn_out = util.MultiHeadAttention(latent_size=params_size, num_heads=1, norm=False, residual=False, cross_type=1, num_latents=self.lat_batch_size, channels=latent_size, name='attn_out')
        self.layer_dense_out_logits = tf.keras.layers.Dense(self.lat_batch_size*params_size, name='dense_out_logits')

        self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        self.net_arch = "{}[inD{}-{:02d}{}{}D{}{}-lat{}x{}-{}]".format(name, inp, net_blocks, ('AT+' if self.net_attn else ''), ('LS+' if self.net_lstm else ''), mid, ('-hds'+str(num_heads) if self.net_attn else ''), latent_spec['num_latents_trans'], latent_size, latent_spec['num_components'])

    def reset_states(self, use_img=False):
        for layer in self.layer_attn: layer.reset_states(use_img=use_img)
        for layer in self.layer_lstm: layer.reset_states()
    def call(self, inputs, store_memory=True, use_img=False, store_real=False, training=None):
        out_accu = [None]*self.net_ins_all
        for i in range(self.net_ins):
            out = tf.cast(inputs['actions'][i], self.compute_dtype)
            if self.layer_attn_in[i] is not None: out = self.layer_attn_in[i](out)
            else: out = self.layer_flatten(out)
            # TODO better to add latents for conditioning/prompting?
            # shape = tf.concat([tf.shape(inputs['obs'])[:1], tf.shape(out)[1:]], axis=0)
            # out = tf.broadcast_to(out, shape)
            out_accu[i] = self.layer_mlp_in[i](out)
        step_size = tf.reshape(tf.cast(inputs['step_size'], self.compute_dtype), [1,1])
        out_accu[-2] = self.layer_step_size_in(step_size)
        out_accu[-1] = tf.cast(inputs['obs'], self.compute_dtype)
        # out = tf.math.add_n(out_accu) # out = tf.math.accumulate_n(out_accu)
        out = tf.concat(out_accu, axis=0)
        
        for i in range(self.net_blocks):
            if self.net_attn: out = tf.squeeze(self.layer_attn[i](tf.expand_dims(out, axis=0), auto_mask=training, store_memory=store_memory, use_img=use_img, store_real=store_real), axis=0)
            if self.net_lstm: out = tf.squeeze(self.layer_lstm[i](tf.expand_dims(out, axis=0), training=training), axis=0)
            out = self.layer_mlp[i](out)

        if self.net_attn_io: out = self.layer_attn_out(out)
        else:
            out = self.layer_flatten(tf.expand_dims(out, axis=0))
            out = self.layer_dense_out_logits(out)
            out = tf.reshape(out, (self.lat_batch_size, -1))

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print('trans net out:', out)
        return out


class GenNet(tf.keras.Model):
    def __init__(self, name, spec_out, force_cont, latent_size, net_blocks=0, net_attn=False, net_lstm=False, net_attn_io=False, num_heads=1, memory_size=None, max_steps=1, force_det_out=False):
        super(GenNet, self).__init__(name=name)
        outp, mid, evo = latent_size*4, latent_size*2, int(latent_size/2)
        self.net_blocks, self.net_attn, self.net_lstm, self.net_attn_io = net_blocks, net_attn, net_lstm, net_attn_io
        self.layer_flatten = tf.keras.layers.Flatten()
        mixture_size = 8 # int(latent_size/2)

        # TODO duplicate per net_outs for better gen?
        self.layer_attn, self.layer_lstm, self.layer_mlp = [], [], []
        for i in range(net_blocks):
            if self.net_attn:
                self.layer_attn += [util.MultiHeadAttention(latent_size=latent_size, num_heads=num_heads, memory_size=memory_size, residual=True, name='attn_{:02d}'.format(i))]
                self.layer_mlp += [util.MLPBlock(hidden_size=mid, latent_size=latent_size, evo=None, residual=True, name='mlp_{:02d}'.format(i))]
            elif self.net_lstm:
                self.layer_lstm += [tf.keras.layers.LSTM(mid, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i))]
                self.layer_mlp += [tf.keras.layers.Dense(latent_size, name='dense_{:02d}'.format(i))]
            else: self.layer_mlp += [util.MLPBlock(hidden_size=mid, latent_size=latent_size, evo=evo, residual=False, name='mlp_{:02d}'.format(i))]

        self.net_outs = len(spec_out); params_size, self.dist, self.logits_step_shape, arch_out = [], [], [], "O"
        for i in range(self.net_outs):
            num_components, event_shape = spec_out[i]['num_components'], spec_out[i]['event_shape']
            if force_det_out:
                params_size += [util.Deterministic.params_size(event_shape)]; self.dist += [util.Deterministic(event_shape)]; typ = 'd'
            elif not force_cont and spec_out[i]['is_discrete']:
                params_size += [util.Categorical.params_size(num_components, event_shape)]; self.dist += [util.Categorical(num_components, event_shape)]; typ = 'c'
            else:
                num_components *= mixture_size
                params_size += [util.MixtureLogistic.params_size(num_components, event_shape)]; self.dist += [util.MixtureLogistic(num_components, event_shape)]; typ = 'mx'
                # params_size += [tfp.layers.MixtureLogistic.params_size(num_components, event_shape)]; self.dist += [tfp.layers.MixtureLogistic(num_components, event_shape)] # makes NaNs
                # params_size += [MixtureMultiNormalTriL.params_size(num_components, event_shape, matrix_size=2)]; self.dist += [MixtureMultiNormalTriL(num_components, event_shape, matrix_size=2)]; typ = 'mt'
            self.logits_step_shape += [tf.TensorShape([1]+[params_size[i]])]
            arch_out += "{}{}".format(typ, num_components)

        self.layer_attn_out, self.layer_mlp_out_logits = [], []
        for i in range(self.net_outs):
            if net_attn_io: self.layer_attn_out += [util.MultiHeadAttention(latent_size=latent_size, num_heads=1, norm=False, residual=False, cross_type=2, num_latents=max_steps, channels=params_size[i], name='attn_out_{:02d}'.format(i))]
            self.layer_mlp_out_logits += [util.MLPBlock(hidden_size=outp, latent_size=params_size[i], evo=evo, residual=False, name='mlp_out_logits_{:02d}'.format(i))]

        self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        self.net_arch = "{}[{:02d}{}{}D{}-{}{}]".format(name, net_blocks, ('AT+' if self.net_attn else ''), ('LS+' if self.net_lstm else ''), mid, arch_out, ('-hds'+str(num_heads) if self.net_attn else ''))

    def reset_states(self, use_img=False):
        for layer in self.layer_attn: layer.reset_states(use_img=use_img)
        for layer in self.layer_lstm: layer.reset_states()
    def call(self, inputs, store_memory=True, use_img=False, store_real=False, batch_size=1, training=None):
        out = tf.cast(inputs['obs'], self.compute_dtype)
        
        for i in range(self.net_blocks):
            if self.net_attn: out = tf.squeeze(self.layer_attn[i](tf.expand_dims(out, axis=0), auto_mask=training, store_memory=store_memory, use_img=use_img, store_real=store_real), axis=0)
            if self.net_lstm: out = tf.squeeze(self.layer_lstm[i](tf.expand_dims(out, axis=0), training=training), axis=0)
            out = self.layer_mlp[i](out)

        if not self.net_attn_io: out = tf.reshape(out, (batch_size, -1))
        out_logits = [None]*self.net_outs
        for i in range(self.net_outs):
            out_logits[i] = out if not self.net_attn_io else self.layer_attn_out[i](out, num_latents=batch_size)
            out_logits[i] = self.layer_mlp_out_logits[i](out_logits[i])

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print('action net out:', out)
        return out_logits


class ValueNet(tf.keras.Model):
    def __init__(self, name, latent_size, net_blocks=0, net_attn=False, net_lstm=False, num_heads=1, memory_size=None):
        super(ValueNet, self).__init__(name=name)
        mid, evo = latent_size*2, int(latent_size/2)
        self.net_blocks, self.net_attn, self.net_lstm = net_blocks, net_attn, net_lstm
        self.layer_flatten = tf.keras.layers.Flatten()

        self.layer_attn, self.layer_lstm, self.layer_mlp = [], [], []
        for i in range(net_blocks):
            if self.net_attn:
                self.layer_attn += [util.MultiHeadAttention(latent_size=latent_size, num_heads=num_heads, memory_size=memory_size, residual=True, name='attn_{:02d}'.format(i))]
                self.layer_mlp += [util.MLPBlock(hidden_size=mid, latent_size=latent_size, evo=None, residual=True, name='mlp_{:02d}'.format(i))]
            elif self.net_lstm:
                self.layer_lstm += [tf.keras.layers.LSTM(mid, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i))]
                self.layer_mlp += [tf.keras.layers.Dense(latent_size, name='dense_{:02d}'.format(i))]
            else: self.layer_mlp += [util.MLPBlock(hidden_size=mid, latent_size=latent_size, evo=evo, residual=False, name='mlp_{:02d}'.format(i))]

        self.layer_dense_out = tf.keras.layers.Dense(1, name='dense_out')

        self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        self.net_arch = "{}[{:02d}{}{}D{}{}]".format(name, net_blocks, ('AT+' if self.net_attn else ''), ('LS+' if self.net_lstm else ''), mid, ('-hds'+str(num_heads) if self.net_attn else ''))

    def reset_states(self, use_img=False):
        for layer in self.layer_attn: layer.reset_states(use_img=use_img)
        for layer in self.layer_lstm: layer.reset_states()
    def call(self, inputs, store_memory=True, use_img=False, store_real=False, training=None):
        out = tf.cast(inputs['obs'], self.compute_dtype)
        
        for i in range(self.net_blocks):
            if self.net_attn: out = tf.squeeze(self.layer_attn[i](tf.expand_dims(out, axis=0), auto_mask=training, store_memory=store_memory, use_img=use_img, store_real=store_real), axis=0)
            if self.net_lstm: out = tf.squeeze(self.layer_lstm[i](tf.expand_dims(out, axis=0), training=training), axis=0)
            out = self.layer_mlp[i](out)

        out = tf.reshape(out, (1, -1))
        out = self.layer_dense_out(out)
        return out


class GeneralAI(tf.keras.Model):
    def __init__(self, arch, env, trader, env_render, max_episodes, max_steps, learn_rate, entropy_contrib, returns_disc, value_cont, force_cont_obs, force_cont_action, latent_size, latent_dist, net_attn_io, aio_max_latents, attn_mem_multi, aug_data_step, aug_data_pos):
        super(GeneralAI, self).__init__()
        compute_dtype = tf.dtypes.as_dtype(self.compute_dtype)
        self.float_min = tf.constant(compute_dtype.min, compute_dtype)
        self.float_maxroot = tf.constant(tf.math.sqrt(compute_dtype.max), compute_dtype)
        self.float_eps = tf.constant(tf.experimental.numpy.finfo(compute_dtype).eps, compute_dtype)
        # self.float_log_min_prob = tf.constant(tf.math.log(self.float_eps), compute_dtype)
        self.attn_img_scales, self.attn_img_max_step_size = int(np.log2(max_steps))+1, tf.constant(1024*64, tf.int32)
        self.compute_zero, self.int32_max, self.int32_maxbit, self.int32_zero, self.float64_zero = tf.constant(0, compute_dtype), tf.constant(tf.int32.max, tf.int32), tf.constant(1073741824, tf.int32), tf.constant(0, tf.int32), tf.constant(0, tf.float64)

        self.arch, self.env, self.trader, self.env_render, self.value_cont, self.force_cont_obs, self.force_cont_action = arch, env, trader, env_render, value_cont, force_cont_obs, force_cont_action
        self.max_episodes, self.max_steps, self.attn_mem_multi, self.entropy_contrib, self.returns_disc = tf.constant(max_episodes, tf.int32), tf.constant(max_steps, tf.int32), tf.constant(attn_mem_multi, tf.int32), tf.constant(entropy_contrib, compute_dtype), tf.constant(returns_disc, tf.float64)
        self.dist_prior = tfp.distributions.Independent(tfp.distributions.Logistic(loc=tf.zeros(latent_size, dtype=self.compute_dtype), scale=10.0), reinterpreted_batch_ndims=1)
        # self.dist_prior = tfp.distributions.Independent(tfp.distributions.Uniform(low=tf.cast(tf.fill(latent_size,-10), dtype=self.compute_dtype), high=10), reinterpreted_batch_ndims=1)

        self.obs_spec, self.obs_zero, _ = util.gym_get_spec(env.observation_space, self.compute_dtype, force_cont=force_cont_obs)
        self.action_spec, _, self.action_zero_out = util.gym_get_spec(env.action_space, self.compute_dtype, force_cont=force_cont_action)
        self.obs_spec_len, self.action_spec_len = len(self.obs_spec), len(self.action_spec)
        self.gym_step_shapes = [feat['step_shape'] for feat in self.obs_spec] + [tf.TensorShape((1,1)), tf.TensorShape((1,1))]
        self.gym_step_dtypes = [feat['dtype'] for feat in self.obs_spec] + [tf.float64, tf.bool]
        self.rewards_zero, self.dones_zero = tf.constant([[0]],tf.float64), tf.constant([[False]],tf.bool)

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

        inputs = {'obs':self.obs_zero, 'rewards':self.rewards_zero, 'dones':self.dones_zero, 'step_size':1}
        if arch in ('PG','AC','TRANS','MU',):
            self.rep = RepNet('RN', self.obs_spec, latent_spec, latent_dist, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, net_attn_io2=net_attn_io2, num_heads=4, memory_size=memory_size, max_steps=max_steps, aug_data_step=aug_data_step, aug_data_pos=aug_data_pos)
            outputs = self.rep(inputs, step=0); rep_dist = self.rep.dist(outputs)
            self.latent_zero = tf.zeros_like(rep_dist.sample(), latent_spec['dtype'])
            inputs['obs'] = self.latent_zero

        # if arch in ('TEST',):
        #     self.gen = GenNet('GN', self.obs_spec, force_cont_obs, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=memory_size, max_steps=max_steps, force_det_out=False); outputs = self.gen(inputs)
        self.action = GenNet('AN', self.action_spec, force_cont_action, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=memory_size, max_steps=max_steps, force_det_out=False); outputs = self.action(inputs)

        if arch in ('AC','MU',):
            if value_cont:
                value_spec = [{'net_type':0, 'dtype':compute_dtype, 'dtype_out':compute_dtype, 'is_discrete':False, 'num_components':8, 'event_shape':(1,), 'step_shape':tf.TensorShape((1,1))}]
                self.value = GenNet('VN', value_spec, False, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=memory_size, max_steps=max_steps, force_det_out=False); outputs = self.value(inputs)
            else: self.value = ValueNet('VN', latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, num_heads=4, memory_size=memory_size); outputs = self.value(inputs)

        if arch in ('TRANS','MU',):
            inputs['actions'] = self.action_zero_out
            # latent_dist = 2; latent_spec = {'dtype':compute_dtype, 'num_latents':lat_batch_size, 'num_latents_trans':lat_batch_size_trans, 'event_shape':(latent_size,), 'num_components':8}
            self.trans = TransNet('TN', self.action_spec, latent_spec, latent_dist, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=memory_size_trans, max_steps=max_steps); outputs = self.trans(inputs)
        if arch in ('MU',):
            reward_spec = [{'net_type':0, 'dtype':tf.float64, 'dtype_out':compute_dtype, 'is_discrete':False, 'num_components':16, 'event_shape':(1,), 'step_shape':tf.TensorShape((1,1))}]
            self.rwd = GenNet('RW', reward_spec, False, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=memory_size, max_steps=max_steps, force_det_out=False); outputs = self.rwd(inputs)
            done_spec = [{'net_type':0, 'dtype':tf.bool, 'dtype_out':tf.int32, 'is_discrete':True, 'num_components':2, 'event_shape':(1,), 'step_shape':tf.TensorShape((1,1))}]
            self.done = GenNet('DO', done_spec, False, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=memory_size, max_steps=max_steps, force_det_out=False); outputs = self.done(inputs)

        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate, epsilon=self.float_eps)
        # self._optimizer = tfa.optimizers.AdamW(learning_rate=learn_rate, weight_decay=1e-7, epsilon=self.float_eps) # _wd7


        metrics_loss = OrderedDict()
        metrics_loss['2rewards*'] = {'rewards_total+':np.float64, 'rewards_final=':np.float64}
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
        for i in range(len(args)): args[i] = args[i].item()
        episode = args[0]
        idx = 1
        for loss_group in self.metrics_loss.values():
            for k in loss_group.keys():
                if k.endswith('='): loss_group[k][episode] = args[idx]
                elif k.endswith('+'): loss_group[k][episode] += args[idx]
                else: loss_group[k][episode] += [args[idx]]
                idx += 1
        return np.asarray(0, np.int32) # dummy


    # TODO use ZMQ for remote messaging, latent pooling
    def env_reset(self, dummy):
        obs, reward, done = self.env.reset(), 0.0, False
        if self.env_render: self.env.render()
        if hasattr(self.env,'np_struc'): rtn = util.gym_struc_to_feat(obs)
        else: rtn = util.gym_space_to_feat(obs, self.env.observation_space)
        rtn += [np.asarray([[reward]], np.float64), np.asarray([[done]], bool)]
        return rtn
    def env_step(self, *args): # args = tuple of ndarrays
        if hasattr(self.env,'np_struc'): action = util.gym_out_to_struc(list(args), self.env.action_dtype)
        else: action = util.gym_out_to_space(args, self.env.action_space, [0])
        obs, reward, done, _ = self.env.step(action)
        if self.env_render: self.env.render()
        if hasattr(self.env,'np_struc'): rtn = util.gym_struc_to_feat(obs)
        else: rtn = util.gym_space_to_feat(obs, self.env.observation_space)
        rtn += [np.asarray([[reward]], np.float64), np.asarray([[done]], bool)]
        return rtn


    def reset_states(self, use_img=False):
        for net in self.layers:
            if hasattr(net, 'reset_states'): net.reset_states(use_img=use_img)


    def loss_diff(self, out, targets=None): # deterministic difference
        if isinstance(out, list):
            loss = self.compute_zero
            for i in range(len(out)):
                o, t = tf.cast(out[i], self.compute_dtype), tf.cast(targets[i], self.compute_dtype)
                loss = loss + tf.math.abs(tf.math.subtract(o, t)) # MAE
                # loss = loss + tf.math.square(tf.math.subtract(o, t)) # MSE
        else:
            out = tf.cast(out, self.compute_dtype)
            if targets is None: diff = out
            else:
                targets = tf.cast(targets, self.compute_dtype)
                diff = tf.math.subtract(out, targets)
            # loss = tf.where(tf.math.less(diff, self.compute_zero), tf.math.negative(diff), diff) # MAE
            loss = tf.math.abs(diff) # MAE
            # loss = tf.math.square(diff) # MSE
        loss = tf.math.reduce_sum(loss, axis=tf.range(1, tf.rank(loss)))
        return loss

    def loss_likelihood(self, dist, targets, probs=False):
        if isinstance(dist, list):
            loss = self.compute_zero
            for i in range(len(dist)):
                t = tf.cast(targets[i], dist[i].dtype)
                if probs: loss = loss - tf.math.exp(dist[i].log_prob(t))
                else: loss = loss - dist[i].log_prob(t)
        else:
            targets = tf.cast(targets, dist.dtype)
            if probs: loss = -tf.math.exp(dist.log_prob(targets))
            else: loss = -dist.log_prob(targets)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)))
        if isinfnan > 0: tf.print('NaN/Inf likelihood loss:', loss)
        return loss

    def loss_bound(self, dist, targets):
        loss = -self.loss_likelihood(dist, targets)
        # if not self.categorical: loss = loss - self.dist_prior.log_prob(targets)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)))
        if isinfnan > 0: tf.print('NaN/Inf bound loss:', loss)
        return loss

    def loss_entropy(self, dist): # "Soft Actor Critic" = try increase entropy
        loss = self.compute_zero
        if self.entropy_contrib > 0.0:
            if isinstance(dist, list):
                for i in range(len(dist)): loss = loss + dist[i].entropy()
            else: loss = dist.entropy()
            loss = -loss * self.entropy_contrib

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)))
        if isinfnan > 0: tf.print('NaN/Inf entropy loss:', loss)
        return loss

    def loss_PG(self, dist, targets, returns, values=None, returns_target=None): # policy gradient, actor/critic
        returns = tf.squeeze(tf.cast(returns, self.compute_dtype), axis=-1)
        loss_lik = self.loss_likelihood(dist, targets, probs=False)
        # loss_lik = loss_lik -self.float_maxroot # -self.float_maxroot, +self.float_log_min_prob, -np.e*17.0, -154.0, -308.0
        if returns_target is not None: returns = tf.squeeze(tf.cast(returns_target, self.compute_dtype), axis=-1) - returns
        if values is not None: returns = returns - tf.squeeze(tf.cast(values, self.compute_dtype), axis=-1)
        loss = loss_lik * returns # / self.float_maxroot

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)))
        if isinfnan > 0: tf.print('NaN/Inf PG loss:', loss)
        return loss



    def PG_actor(self, inputs):
        print("tracing -> GeneralAI PG_actor")
        obs, actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): obs[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['event_shape'])
        for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
        rewards = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        dones = tf.TensorArray(tf.bool, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        returns = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))

        step = tf.constant(0)
        # while step < self.max_steps and not inputs['dones'][-1][0]:
        while not inputs['dones'][-1][0]:
            for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])

            rep_logits = self.rep(inputs, step=step); rep_dist = self.rep.dist(rep_logits)
            inputs['obs'] = rep_dist.sample()

            action_logits = self.action(inputs)
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                action_dist = self.action.dist[i](action_logits[i])
                action[i] = action_dist.sample()
                actions[i] = actions[i].write(step, action[i][-1])
                action[i] = util.discretize(action[i], self.action_spec[i], self.force_cont_action)

            np_in = tf.numpy_function(self.env_step, action, self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs['obs'], inputs['rewards'], inputs['dones'] = np_in[:-2], np_in[-2], np_in[-1]

            rewards = rewards.write(step, inputs['rewards'][-1])
            dones = dones.write(step, inputs['dones'][-1])
            returns = returns.write(step, [self.float64_zero])
            returns_updt = returns.stack()
            returns_updt = returns_updt + inputs['rewards'][-1]
            returns = returns.unstack(returns_updt)

            step += 1

        outputs = {}
        out_obs, out_actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): out_obs[i] = obs[i].stack()
        for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
        outputs['obs'], outputs['actions'], outputs['rewards'], outputs['dones'], outputs['returns'] = out_obs, out_actions, rewards.stack(), dones.stack(), returns.stack()
        return outputs, inputs

    # def PG_learner(self, inputs, training=True):
    #     print("tracing -> GeneralAI PG_learner")
    #     loss = {}

    #     with tf.GradientTape() as tape:
    #         batch_size = tf.shape(inputs['rewards'])[0]
    #         rep_logits = self.rep(inputs, training=training); rep_dist = self.rep.dist(rep_logits)
    #         inputs['obs'] = rep_dist.sample()

    #         action_logits = self.action(inputs, batch_size=batch_size, training=training)
    #         action_dist = [None]*self.action_spec_len
    #         for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
    #         loss['action'] = self.loss_PG(action_dist, inputs['actions'], inputs['returns'])

    #     gradients = tape.gradient(loss['action'], self.rep.trainable_variables + self.action.trainable_variables)
    #     self._optimizer.apply_gradients(zip(gradients, self.rep.trainable_variables + self.action.trainable_variables))

    #     # loss['entropy'] = self.loss_entropy(action_dist)
    #     return loss

    def PG_learner_onestep(self, inputs, training=True):
        print("tracing -> GeneralAI PG_learner_onestep")
        loss = {}
        loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        for step in tf.range(tf.shape(inputs['dones'])[0]):
            obs = [None]*self.obs_spec_len
            for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
            returns = inputs['returns'][step:step+1]

            inputs_step = {'obs':obs}
            with tf.GradientTape(persistent=True) as tape_action:
                rep_logits = self.rep(inputs_step, step=step); rep_dist = self.rep.dist(rep_logits)
                inputs_step['obs'] = rep_dist.sample()

            with tape_action:
                action_logits = self.action(inputs_step)
                action_dist = [None]*self.action_spec_len
                for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
                loss_action = self.loss_PG(action_dist, action, returns)
            gradients = tape_action.gradient(loss_action, self.rep.trainable_variables + self.action.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.rep.trainable_variables + self.action.trainable_variables))
            loss_actions = loss_actions.write(step, loss_action)

        loss['action'] = loss_actions.concat()
        return loss

    def PG_run_episode(self, inputs, episode, training=True):
        print("tracing -> GeneralAI PG_run_episode")
        # TODO how unlimited length episodes without sacrificing returns signal?
        while not inputs['dones'][-1][0]:
            self.reset_states(); outputs, inputs = self.PG_actor(inputs)
            self.reset_states(); loss = self.PG_learner_onestep(outputs)

            metrics = [episode, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                tf.math.reduce_mean(loss['action']), tf.math.reduce_mean(outputs['returns'])]
            if self.trader: metrics += [tf.math.reduce_mean(tf.concat([outputs['obs'][3],inputs['obs'][3]],0)), inputs['obs'][3][-1][0],
                tf.math.reduce_mean(tf.concat([outputs['obs'][4],inputs['obs'][4]],0)), tf.math.reduce_mean(tf.concat([outputs['obs'][5],inputs['obs'][5]],0)),
                inputs['obs'][0][-1][0] - outputs['obs'][0][0][0],]
            dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

    def PG(self):
        print("tracing -> GeneralAI PG")
        for episode in tf.range(self.max_episodes):
            tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}
            self.PG_run_episode(inputs, episode)



    def AC_actor(self, inputs):
        print("tracing -> GeneralAI AC_actor")
        obs, actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): obs[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['event_shape'])
        for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
        rewards = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        dones = tf.TensorArray(tf.bool, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        returns = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        step = tf.constant(0)
        while step < self.max_steps and not inputs['dones'][-1][0]: # max_steps for limiting memory usage
            # tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            # tf.autograph.experimental.set_loop_options(shape_invariants=[(inputs['obs'], [tf.TensorShape([None,None])]), (inputs['rewards'], tf.TensorShape([None,None])), (inputs['dones'], tf.TensorShape([None,None]))])
            # tf.autograph.experimental.set_loop_options(shape_invariants=[(outputs['rewards'], [None,1]), (outputs['dones'], [None,1]), (outputs['returns'], [None,1])])
            for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])

            rep_logits = self.rep(inputs, step=step); rep_dist = self.rep.dist(rep_logits)
            inputs['obs'] = rep_dist.sample()

            action_logits = self.action(inputs)
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                action_dist = self.action.dist[i](action_logits[i])
                action[i] = action_dist.sample()
                actions[i] = actions[i].write(step, action[i][-1])
                action[i] = util.discretize(action[i], self.action_spec[i], self.force_cont_action)

            np_in = tf.numpy_function(self.env_step, action, self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            # inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}
            inputs['obs'], inputs['rewards'], inputs['dones'] = np_in[:-2], np_in[-2], np_in[-1]

            rewards = rewards.write(step, inputs['rewards'][-1])
            dones = dones.write(step, inputs['dones'][-1])
            returns = returns.write(step, [self.float64_zero])
            returns_updt = returns.stack()
            returns_updt = returns_updt + inputs['rewards'][-1]
            returns = returns.unstack(returns_updt)

            step += 1

        # for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1]) # TD loss

        outputs = {}
        out_obs, out_actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): out_obs[i] = obs[i].stack()
        for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
        outputs['obs'], outputs['actions'], outputs['rewards'], outputs['dones'], outputs['returns'] = out_obs, out_actions, rewards.stack(), dones.stack(), returns.stack()
        return outputs, inputs

    # def AC_learner(self, inputs, training=True):
    #     print("tracing -> GeneralAI AC_learner")
    #     loss = {}

    #     inputs_lat = {'obs':self.latent_zero}
    #     with tf.GradientTape(persistent=True) as tape_value, tf.GradientTape(persistent=True) as tape_action:
    #         batch_size = tf.shape(inputs['rewards'])[0]
    #         rep_logits = self.rep(inputs, training=training); rep_dist = self.rep.dist(rep_logits)
    #         inputs_lat['obs'] = rep_dist.sample()

    #     returns = inputs['returns']
    #     # with tf.GradientTape() as tape_value:
    #     #     rep_logits = self.rep(inputs, training=training); rep_dist = self.rep.dist(rep_logits)
    #     #     inputs_lat['obs'] = rep_dist.sample()
    #     with tape_value:
    #         if self.value_cont:
    #             value_logits = self.value(inputs_lat, training=training); value_dist = self.value.dist[0](value_logits[0])
    #             loss['value'] = self.loss_likelihood(value_dist, returns)
    #             values = value_dist.sample()
    #         else:
    #             values = self.value(inputs_lat, training=training)
    #             loss['value'] = self.loss_diff(values, returns)
    #     gradients = tape_value.gradient(loss['value'], self.rep.trainable_variables + self.value.trainable_variables)
    #     self._optimizer.apply_gradients(zip(gradients, self.rep.trainable_variables + self.value.trainable_variables))

    #     returns = tf.cast(returns, self.compute_dtype)
    #     # with tf.GradientTape() as tape_action:
    #     #     rep_logits = self.rep(inputs, training=training); rep_dist = self.rep.dist(rep_logits)
    #     #     inputs_lat['obs'] = rep_dist.sample()
    #     with tape_action:
    #         advantages = returns - values # new chance of chosen action: if over predict = push away, if under predict = push closer, if can predict = stay
    #         action_logits = self.action(inputs_lat, batch_size=batch_size, training=training)
    #         action_dist = [None]*self.action_spec_len
    #         for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
    #         loss['action'] = self.loss_PG(action_dist, inputs['actions'], advantages)
    #     gradients = tape_action.gradient(loss['action'], self.rep.trainable_variables + self.action.trainable_variables)
    #     self._optimizer.apply_gradients(zip(gradients, self.rep.trainable_variables + self.action.trainable_variables))

    #     # loss['entropy'] = self.loss_entropy(action_dist)
    #     loss['advantages'] = advantages
    #     return loss

    def AC_learner_onestep(self, inputs, training=True):
        print("tracing -> GeneralAI AC_learner_onestep")
        loss = {}
        loss_returns = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        metric_advantages = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        # return_goal = tf.constant([[200.0]], tf.float64)
        for step in tf.range(tf.shape(inputs['dones'])[0]):
            obs = [None]*self.obs_spec_len
            for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
            returns = inputs['returns'][step:step+1]

            inputs_step = {'obs':obs}
            with tf.GradientTape(persistent=True) as tape_value, tf.GradientTape(persistent=True) as tape_action:
                rep_logits = self.rep(inputs_step, step=step); rep_dist = self.rep.dist(rep_logits)
                inputs_step['obs'] = rep_dist.sample()

            with tape_value:
                if self.value_cont:
                    value_logits = self.value(inputs_step); value_dist = self.value.dist[0](value_logits[0])
                    loss_return = self.loss_likelihood(value_dist, returns)
                    values = value_dist.sample()
                else:
                    values = self.value(inputs_step)
                    loss_return = self.loss_diff(values, returns)
            gradients = tape_value.gradient(loss_return, self.rep.trainable_variables + self.value.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.rep.trainable_variables + self.value.trainable_variables))
            loss_returns = loss_returns.write(step, loss_return)

            with tape_action:
                action_logits = self.action(inputs_step)
                action_dist = [None]*self.action_spec_len
                for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
                loss_action = self.loss_PG(action_dist, action, returns, values)
                # loss_action = self.loss_PG(action_dist, action, returns, values, returns_target=return_goal) # lPGt
                # loss_action = self.loss_PG(action_dist, action, loss_return) # lPGv
            gradients = tape_action.gradient(loss_action, self.rep.trainable_variables + self.action.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.rep.trainable_variables + self.action.trainable_variables))
            loss_actions = loss_actions.write(step, loss_action)
            metric_advantages = metric_advantages.write(step, (returns - tf.cast(values,tf.float64))[0])

            # return_goal -= inputs['rewards'][step:step+1]; return_goal.set_shape((1,1))

        loss['value'], loss['action'] = loss_returns.concat(), loss_actions.concat()
        loss['advantages'] = metric_advantages.concat()
        return loss

    def AC_run_episode(self, inputs, episode, training=True):
        print("tracing -> GeneralAI AC_run_episode")
        while not inputs['dones'][-1][0]:
            # tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            outputs, inputs = self.AC_actor(inputs)
            loss = self.AC_learner_onestep(outputs)

            metrics = [episode, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                tf.math.reduce_mean(loss['action']), tf.math.reduce_mean(loss['value']),
                tf.math.reduce_mean(outputs['returns']), tf.math.reduce_mean(loss['advantages']),
            ]
            if self.trader: metrics += [tf.math.reduce_mean(tf.concat([outputs['obs'][3],inputs['obs'][3]],0)), inputs['obs'][3][-1][0],
                tf.math.reduce_mean(tf.concat([outputs['obs'][4],inputs['obs'][4]],0)), tf.math.reduce_mean(tf.concat([outputs['obs'][5],inputs['obs'][5]],0)),
                inputs['obs'][0][-1][0] - outputs['obs'][0][0][0],]
            dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

    def AC(self):
        print("tracing -> GeneralAI AC")
        for episode in tf.range(self.max_episodes):
            tf.autograph.experimental.set_loop_options(parallel_iterations=1) # TODO parallel wont work with single instance env, will this work multiple?
            self.reset_states()
            np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}
            self.AC_run_episode(inputs, episode)



    def TRANS_actor(self, inputs):
        print("tracing -> GeneralAI TRANS_actor")
        obs = [None]*self.obs_spec_len
        for i in range(self.obs_spec_len): obs[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['event_shape'])
        actions = [None]*self.action_spec_len
        for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
        # latents_next = tf.TensorArray(self.latent_spec['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['event_shape'])
        rewards = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        targets = [None]*self.obs_spec_len
        for i in range(self.obs_spec_len): targets[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['event_shape'])

        # inputs_rep = {'obs':self.latent_zero}
        inputs_rep = {'obs':self.latent_zero, 'actions':self.action_zero_out, 'step_size':1}
        step = tf.constant(0)
        while not inputs['dones'][-1][0]:
            for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])
            for i in range(self.action_spec_len): actions[i] = actions[i].write(step, inputs_rep['actions'][i][-1])

            rep_logits = self.rep(inputs, step=step); rep_dist = self.rep.dist(rep_logits)
            inputs_rep['obs'] = rep_dist.sample()

            trans_logits = self.trans(inputs_rep); trans_dist = self.trans.dist(trans_logits)
            inputs_rep['obs'] = trans_dist.sample()
            # latents_next = latents_next.write(step, inputs_rep['obs'][-1])

            action_logits = self.action(inputs_rep)
            action, action_dis = [None]*self.action_spec_len, [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                action_dist = self.action.dist[i](action_logits[i])
                action[i] = action_dist.sample()
                # action[i] = tf.cast(action[i], self.action_spec[i]['dtype_out']) # force_det_out
                action_dis[i] = util.discretize(action[i], self.action_spec[i], self.force_cont_action)
            inputs_rep['actions'] = action

            np_in = tf.numpy_function(self.env_step, action_dis, self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs['obs'], inputs['rewards'], inputs['dones'] = np_in[:-2], np_in[-2], np_in[-1]

            rewards = rewards.write(step, inputs['rewards'][-1])
            for i in range(self.obs_spec_len): targets[i] = targets[i].write(step, inputs['obs'][i][-1])
            step += 1

        outputs = {}
        out_obs = [None]*self.obs_spec_len
        for i in range(self.obs_spec_len): out_obs[i] = obs[i].stack()
        out_actions = [None]*self.action_spec_len
        for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
        out_targets = [None]*self.obs_spec_len
        for i in range(self.obs_spec_len): out_targets[i] = targets[i].stack()
        # outputs['obs'], outputs['rewards'], outputs['targets'] = latents_next.stack(), rewards.stack(), out_targets
        # outputs['obs'], outputs['rewards'], outputs['targets'] = out_obs, rewards.stack(), out_targets
        outputs['obs'], outputs['actions'], outputs['rewards'], outputs['targets'] = out_obs, out_actions, rewards.stack(), out_targets
        return outputs, inputs

    # def TRANS_learner(self, inputs, training=True):
    #     print("tracing -> GeneralAI TRANS_learner")
    #     loss = {}

    #     # inputs_rep = {'obs':self.latent_zero}
    #     inputs_rep = {'obs':self.latent_zero, 'actions':inputs['actions']}
    #     with tf.GradientTape() as tape:
    #         batch_size = tf.shape(inputs['rewards'])[0]
    #         rep_logits = self.rep(inputs, training=training); rep_dist = self.rep.dist(rep_logits)
    #         inputs_rep['obs'] = rep_dist.sample()

    #         trans_logits = self.trans(inputs_rep, training=training); trans_dist = self.trans.dist(trans_logits)
    #         inputs_rep['obs'] = trans_dist.sample()
    #         # latents_next = latents_next.write(step, inputs_rep['obs'][-1])

    #         action_logits = self.action(inputs_rep, batch_size=batch_size, training=training)
    #         action_dist = [None]*self.action_spec_len
    #         for i in range(self.action_spec_len):
    #             action_dist[i] = self.action.dist[i](action_logits[i])
    #             # action_dist[i] = action_dist[i].sample() # force_det_out
    #         loss['action'] = self.loss_likelihood(action_dist, inputs['targets'])
    #         # loss['action'] = self.loss_diff(action_dist, inputs['targets']) # force_det_out

    #     gradients = tape.gradient(loss['action'], self.rep.trainable_variables + self.trans.trainable_variables + self.action.trainable_variables)
    #     self._optimizer.apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.action.trainable_variables))

    #     return loss

    def TRANS_learner_onestep(self, inputs, training=True):
        print("tracing -> GeneralAI TRANS_learner_onestep")
        loss = {}
        loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        for step in tf.range(tf.shape(inputs['rewards'])[0]):
            obs = [None]*self.obs_spec_len
            for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
            targets = [None]*self.action_spec_len
            for i in range(self.action_spec_len): targets[i] = inputs['targets'][i][step:step+1]; targets[i].set_shape(self.action_spec[i]['step_shape'])

            inputs_step = {'obs':obs, 'actions':action, 'step_size':1}
            with tf.GradientTape(persistent=True) as tape_action:
                rep_logits = self.rep(inputs_step, step=step); rep_dist = self.rep.dist(rep_logits)
                inputs_step['obs'] = rep_dist.sample()

            with tape_action:
                trans_logits = self.trans(inputs_step); trans_dist = self.trans.dist(trans_logits)
                inputs_step['obs'] = trans_dist.sample()

                action_logits = self.action(inputs_step)
                action_dist = [None]*self.action_spec_len
                for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
                loss_action = self.loss_likelihood(action_dist, targets)
            gradients = tape_action.gradient(loss_action, self.rep.trainable_variables + self.trans.trainable_variables + self.action.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.action.trainable_variables))
            loss_actions = loss_actions.write(step, loss_action)

        loss['action'] = loss_actions.concat()
        return loss

    def TRANS_run_episode(self, inputs, episode, training=True):
        print("tracing -> GeneralAI TRANS_run_episode")
        while not inputs['dones'][-1][0]:
            self.reset_states(); outputs, inputs = self.TRANS_actor(inputs)
            self.reset_states(); loss = self.TRANS_learner_onestep(outputs)

            metrics = [episode, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                tf.math.reduce_mean(loss['action'])]
            dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

    def TRANS(self):
        print("tracing -> GeneralAI TRANS")
        for episode in tf.range(self.max_episodes):
            tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}
            self.TRANS_run_episode(inputs, episode)




def params(): pass
load_model, save_model = False, False
max_episodes = 1000
learn_rate = 1e-6 # 5 = testing, 6 = more stable/slower
entropy_contrib = 0 # 1e-8
returns_disc = 1.0
value_cont = True
force_cont_obs, force_cont_action = False, False
latent_size = 128
latent_dist = 0 # 0 = deterministic, 1 = categorical, 2 = continuous
net_attn_io = True
aio_max_latents = 32
attn_mem_multi = 2 # attn_img_base
aug_data_step, aug_data_pos = True, False

device_type = 'GPU' # use GPU for large networks (over 8 total net blocks?) or output data (512 bytes?)
device_type = 'CPU'

machine, device, extra = 'dev', 0, '' # _prs2 _wd7 _lPGv _RfB _train _entropy3 _mae _perO-NR-NT-G-Nrez _rez-rezoR-rezoT-rezoG _mixlog-abs-log1p-Nreparam _obs-tsBoxF-dataBoxI_round _Nexp-Ne9-Nefmp36-Nefmer154-Nefme308-emr-Ndiv _MUimg-entropy-values-policy-Netoe _AC-Nonestep-aing _mem-sort _stepE _cncat

trader, env_async, env_async_clock, env_async_speed = False, False, 0.001, 160.0
env_name, max_steps, env_render, env = 'CartPole', 256, False, gym.make('CartPole-v0') # ; env.observation_space.dtype = np.dtype('float64')
# env_name, max_steps, env_render, env = 'CartPole', 512, False, gym.make('CartPole-v1') # ; env.observation_space.dtype = np.dtype('float64')
# env_name, max_steps, env_render, env = 'LunarLand', 1024, False, gym.make('LunarLander-v2')
# env_name, max_steps, env_render, env = 'Copy', 256, False, gym.make('Copy-v0') # DuplicatedInput-v0 RepeatCopy-v0 Reverse-v0 ReversedAddition-v0 ReversedAddition3-v0
# env_name, max_steps, env_render, env = 'ProcgenChaser', 1024, False, gym.make('procgen-chaser-v0')
# env_name, max_steps, env_render, env = 'ProcgenMiner', 1024, False, gym.make('procgen-miner-v0')
# env_name, max_steps, env_render, env = 'Tetris', 22528, False, gym.make('ALE/Tetris-v5') # max_steps 21600

# env_name, max_steps, env_render, env = 'CartPoleCont', 256, False, gym.make('CartPoleContinuousBulletEnv-v0'); env.observation_space.dtype = np.dtype('float64')
# env_name, max_steps, env_render, env = 'LunarLandCont', 1024, False, gym.make('LunarLanderContinuous-v2') # max_steps 1000
# import envs_local.bipedal_walker as env_; env_name, max_steps, env_render, env = 'BipedalWalker', 2048, False, env_.BipedalWalker() # max_steps 1600
# env_name, max_steps, env_render, env = 'Hopper', 1024, False, gym.make('HopperBulletEnv-v0') # max_steps 1000

# from pettingzoo.butterfly import pistonball_v4; env_name, max_steps, env_render, env = 'PistonBall', 1, False, pistonball_v4.env()

# import envs_local.random_env as env_; env_name, max_steps, env_render, env = 'TestRnd', 16, False, env_.RandomEnv(True)
# import envs_local.data_env as env_; env_name, max_steps, env_render, env = 'DataShkspr', 64, False, env_.DataEnv('shkspr')
# # import envs_local.data_env as env_; env_name, max_steps, env_render, env = 'DataMnist', 64, False, env_.DataEnv('mnist')
# import gym_trader; tenv = 2; env_name, max_steps, env_render, env, trader = 'Trader'+str(tenv), 1024*2, False, gym.make('Trader-v0', agent_id=device, env=tenv), True

# max_steps = 32 # max replay buffer or train interval or bootstrap

# arch = 'TEST' # testing architechures
# arch = 'PG' # Policy Gradient agent, PG loss
arch = 'AC' # Actor Critic, PG and advantage loss
# arch = 'TRANS' # learned Transition dynamics, autoregressive likelihood loss
# arch = 'MU' # Dreamer/planner w/imagination (DeepMind MuZero)
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
            if load_model and tf.io.gfile.exists(model_file):
                net.load_weights(model_file, by_name=True, skip_mismatch=True)
                print("LOADED {} weights from {}".format(net.name, model_file)); loaded_model = True
            name_arch += "   {}{}".format(net.net_arch, 'load' if loaded_model else 'new')


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

        name = "{}-{}-a{}-{}{}".format(name, machine, device, time.strftime("%y-%m-%d-%H-%M-%S"), extra)
        total_steps = int(np.sum(metrics_loss['1steps']['steps+']))
        step_time = total_time/total_steps
        title = "{}    [{}-{}] {}\ntime:{}    steps:{}    t/s:{:.8f}".format(name, device_type, tf.keras.backend.floatx(), name_arch, util.print_time(total_time), total_steps, step_time)
        title += "     |     lr:{}    dis:{}    en:{}    al:{}    am:{}    ms:{}".format(learn_rate, returns_disc, entropy_contrib, aio_max_latents, attn_mem_multi, max_steps)
        title += "     |     a-clk:{}    a-spd:{}    aug:{}{}    aio:{}".format(env_async_clock, env_async_speed, ('S' if aug_data_step else ''), ('P' if aug_data_pos else ''), ('Y' if net_attn_io else 'N')); print(title)

        import matplotlib as mpl
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue','lightblue','green','lime','red','lavender','turquoise','cyan','magenta','salmon','yellow','gold','black','brown','purple','pink','orange','teal','coral','darkgreen','tan'])
        plt.figure(num=name, figsize=(34, 18), tight_layout=True)
        xrng, i, vplts, lim = np.arange(0, max_episodes, 1), 0, 0, 0.03
        for loss_group_name in metrics_loss.keys(): vplts += int(loss_group_name[0])

        for loss_group_name, loss_group in metrics_loss.items():
            rows, col, m_min, m_max, combine = int(loss_group_name[0]), 0, [0]*len(loss_group), [0]*len(loss_group), loss_group_name.endswith('*')
            if combine: spg = plt.subplot2grid((vplts, 1), (i, 0), rowspan=rows, xlim=(0, max_episodes)); plt.grid(axis='y',alpha=0.3)
            for metric_name, metric in loss_group.items():
                metric = np.asarray(metric, np.float64); m_min[col], m_max[col] = np.nanquantile(metric, lim), np.nanquantile(metric, 1.0-lim)
                if not combine: spg = plt.subplot2grid((vplts, len(loss_group)), (i, col), rowspan=rows, xlim=(0, max_episodes), ylim=(m_min[col], m_max[col])); plt.grid(axis='y',alpha=0.3)
                # plt.plot(xrng, talib.EMA(metric, timeperiod=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
                # plt.plot(xrng, bottleneck.move_mean(metric, window=max_episodes//10+2, min_count=1), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
                plt.plot(xrng, util.ewma(metric, window=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
                plt.ylabel('value'); plt.xlabel('episode'); plt.legend(loc='upper left'); col+=1
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
