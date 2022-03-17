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
    def __init__(self, name, inputs, opt_spec, spec_in, latent_spec, latent_dist, latent_size, net_blocks=0, net_attn=False, net_lstm=False, net_attn_io=False, net_attn_io2=False, num_heads=1, memory_size=None, aug_data_step=False, aug_data_pos=False):
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

        self.optimizer = OrderedDict()
        for spec in opt_spec: self.optimizer[spec['name']] = util.optimizer(name, spec)

        self(inputs); self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        self.net_arch = "{}[inD{}-{:02d}{}{}D{}{}-lat{}x{}-{}]".format(name, inp, net_blocks, ('AT+' if self.net_attn else ''), ('LS+' if self.net_lstm else ''), mid, ('-hds'+str(num_heads) if self.net_attn else ''), latent_spec['num_latents'], latent_size, latent_spec['num_components'])

    def reset_states(self, use_img=False):
        for layer in self.layer_attn: layer.reset_states(use_img=use_img)
        for layer in self.layer_lstm: layer.reset_states()
    def call(self, inputs, step=tf.constant(0), store_memory=True, use_img=False, store_real=False, training=None):
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
    def __init__(self, name, inputs, opt_spec, spec_in, latent_spec, latent_dist, latent_size, net_blocks=0, net_attn=False, net_lstm=False, net_attn_io=False, num_heads=1, memory_size=None): # spec_in=[] for no action conditioning
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
        else: self.layer_dense_out_logits = tf.keras.layers.Dense(self.lat_batch_size*params_size, name='dense_out_logits')

        self.optimizer = OrderedDict()
        for spec in opt_spec: self.optimizer[spec['name']] = util.optimizer(name, spec)

        self(inputs); self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
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
    def __init__(self, name, inputs, opt_spec, spec_out, force_cont, latent_size, net_blocks=0, net_attn=False, net_lstm=False, net_attn_io=False, num_heads=1, memory_size=None, max_steps=1, force_det_out=False):
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
            # if net_attn_io: self.layer_attn_out += [util.MultiHeadAttention(latent_size=latent_size, num_heads=1, norm=False, residual=False, cross_type=2, num_latents=max_steps, channels=params_size[i], name='attn_out_{:02d}'.format(i))]
            self.layer_mlp_out_logits += [util.MLPBlock(hidden_size=outp, latent_size=params_size[i], evo=evo, residual=False, name='mlp_out_logits_{:02d}'.format(i))]

        self.optimizer = OrderedDict()
        for spec in opt_spec: self.optimizer[spec['name']] = util.optimizer(name, spec)

        self.stats_rwd = {'b1':tf.constant(0.99,tf.float64), 'b1_n':tf.constant(0.01,tf.float64), 'b2':tf.constant(0.99,tf.float64), 'b2_n':tf.constant(0.01,tf.float64),
            'ma':tf.Variable(0, dtype=tf.float64, trainable=False, name='{}/stats_rwd/ma'.format(name)), 'ema':tf.Variable(0, dtype=tf.float64, trainable=False, name='{}/stats_rwd/ema'.format(name)), 'iter':tf.Variable(0, dtype=tf.float64, trainable=False, name='{}/stats_rwd/iter'.format(name)),}
        self.stats_loss = {'b1':tf.constant(0.99,self.compute_dtype), 'b1_n':tf.constant(0.01,self.compute_dtype), 'b2':tf.constant(0.99,self.compute_dtype), 'b2_n':tf.constant(0.01,self.compute_dtype),
            'ma':tf.Variable(0, dtype=self.compute_dtype, trainable=False, name='{}/stats_loss/ma'.format(name)), 'ema':tf.Variable(0, dtype=self.compute_dtype, trainable=False, name='{}/stats_loss/ema'.format(name)), 'iter':tf.Variable(0, dtype=self.compute_dtype, trainable=False, name='{}/stats_loss/iter'.format(name)),}

        self(inputs); self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        self.net_arch = "{}[{:02d}{}{}D{}-{}{}]".format(name, net_blocks, ('AT+' if self.net_attn else ''), ('LS+' if self.net_lstm else ''), mid, arch_out, ('-hds'+str(num_heads) if self.net_attn else ''))

    def reset_states(self, use_img=False):
        for layer in self.layer_attn: layer.reset_states(use_img=use_img)
        for layer in self.layer_lstm: layer.reset_states()
    def call(self, inputs, batch_size=tf.constant(1), store_memory=True, use_img=False, store_real=False, training=None):
        out = tf.cast(inputs['obs'], self.compute_dtype)

        for i in range(self.net_blocks):
            if self.net_attn: out = tf.squeeze(self.layer_attn[i](tf.expand_dims(out, axis=0), auto_mask=training, store_memory=store_memory, use_img=use_img, store_real=store_real), axis=0)
            if self.net_lstm: out = tf.squeeze(self.layer_lstm[i](tf.expand_dims(out, axis=0), training=training), axis=0)
            out = self.layer_mlp[i](out)

        # if not self.net_attn_io: out = tf.reshape(out, (batch_size, -1))
        out = tf.reshape(out, (batch_size, -1))
        out_logits = [None]*self.net_outs
        for i in range(self.net_outs):
            # out_logits[i] = out if not self.net_attn_io else self.layer_attn_out[i](out, num_latents=batch_size)
            # out_logits[i] = self.layer_mlp_out_logits[i](out_logits[i])
            out_logits[i] = self.layer_mlp_out_logits[i](out)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print('action net out:', out)
        return out_logits


class ValueNet(tf.keras.Model):
    def __init__(self, name, inputs, opt_spec, latent_size, net_blocks=0, net_attn=False, net_lstm=False, num_heads=1, memory_size=None):
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

        self.optimizer = OrderedDict()
        for spec in opt_spec: self.optimizer[spec['name']] = util.optimizer(name, spec)

        self(inputs); self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
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
        self.float64_eps = tf.constant(tf.experimental.numpy.finfo(tf.float64).eps, tf.float64)
        # self.float_log_min_prob = tf.constant(tf.math.log(self.float_eps), compute_dtype)
        self.compute_zero, self.int32_max, self.int32_maxbit, self.int32_zero, self.float64_zero = tf.constant(0, compute_dtype), tf.constant(tf.int32.max, tf.int32), tf.constant(1073741824, tf.int32), tf.constant(0, tf.int32), tf.constant(0, tf.float64)

        self.arch, self.env, self.trader, self.env_render, self.value_cont, self.force_cont_obs, self.force_cont_action = arch, env, trader, env_render, value_cont, force_cont_obs, force_cont_action
        self.max_episodes, self.max_steps, self.learn_rate, self.attn_mem_multi, self.entropy_contrib, self.returns_disc = tf.constant(max_episodes, tf.int32), tf.constant(max_steps, tf.int32), tf.constant(learn_rate, tf.float64), tf.constant(attn_mem_multi, tf.int32), tf.constant(entropy_contrib, compute_dtype), tf.constant(returns_disc, tf.float64)
        self.dist_prior = tfp.distributions.Independent(tfp.distributions.Logistic(loc=tf.zeros(latent_size, dtype=self.compute_dtype), scale=10.0), reinterpreted_batch_ndims=1)
        # self.dist_prior = tfp.distributions.Independent(tfp.distributions.Uniform(low=tf.cast(tf.fill(latent_size,-10), dtype=self.compute_dtype), high=10), reinterpreted_batch_ndims=1)
        self.initializer = tf.keras.initializers.GlorotUniform()

        self.obs_spec, self.obs_zero, _ = util.gym_get_spec(env.observation_space, self.compute_dtype, force_cont=force_cont_obs)
        self.action_spec, _, self.action_zero_out = util.gym_get_spec(env.action_space, self.compute_dtype, force_cont=force_cont_action)
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

        inputs = {'obs':self.obs_zero, 'rewards':self.rewards_zero, 'dones':self.dones_zero, 'step_size':1}
        if arch in ('PG','AC','TRANS','MU','VPN','SPR','MU2','MU3','MU4',):
            opt_spec = [
                {'name':'act', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps},
                {'name':'PG', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps},
                {'name':'PGL', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps},
                {'name':'trans', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps},
                {'name':'rwd', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps},
                {'name':'done', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps},
            ]
            self.rep = RepNet('RN', inputs, opt_spec, self.obs_spec, latent_spec, latent_dist, latent_size, net_blocks=0, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, net_attn_io2=net_attn_io2, num_heads=4, memory_size=memory_size, aug_data_step=aug_data_step, aug_data_pos=aug_data_pos)
            outputs = self.rep(inputs, step=0); rep_dist = self.rep.dist(outputs)
            self.latent_zero = tf.zeros_like(rep_dist.sample(), latent_spec['dtype'])
            inputs['obs'] = self.latent_zero
            self.rep.optimizer_weights = []
            for spec in opt_spec: self.rep.optimizer_weights += util.optimizer_build(self.rep.optimizer[spec['name']], self.rep.trainable_variables)
            util.net_build(self.rep, self.initializer)

        opt_spec = [{'name':'action', 'type':'ar', 'schedule_type':'', 'learn_rate':tf.constant(6e-4, tf.float64), 'float_eps':self.float_eps}]
        self.action = GenNet('AN', inputs, opt_spec, self.action_spec, force_cont_action, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=memory_size, max_steps=max_steps, force_det_out=False); outputs = self.action(inputs)
        self.action.optimizer_weights = util.optimizer_build(self.action.optimizer['action'], self.action.trainable_variables)
        util.net_build(self.action, self.initializer)
        if arch in ('MU4',):
            opt_spec = [{'name':'action', 'type':'ar', 'schedule_type':'', 'learn_rate':tf.constant(2e-9, tf.float64), 'float_eps':self.float_eps}]
            self.actionL = GenNet('ANL', inputs, opt_spec, self.action_spec, force_cont_action, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=memory_size, max_steps=max_steps, force_det_out=False); outputs = self.actionL(inputs)
            self.actionL.optimizer_weights = util.optimizer_build(self.actionL.optimizer['action'], self.actionL.trainable_variables)
            util.net_build(self.actionL, self.initializer)
        if arch in ('MU4',):
            memory_size_actin = (lat_batch_size+2) * max_steps # return_goal and step_size
            inputs['actions'] = [tf.constant([[0]],tf.float64)]
            query_spec = [{'net_type':0, 'dtype':tf.float64, 'dtype_out':compute_dtype, 'is_discrete':False, 'num_components':1, 'event_shape':(1,), 'event_size':1, 'channels':1, 'step_shape':tf.TensorShape((1,1)), 'num_latents':1}]
            self.actin = TransNet('ACT1', inputs, [], query_spec, latent_spec, latent_dist, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=memory_size_actin); outputs = self.actin(inputs)
            opt_spec = [{'name':'act', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps}]
            self.actout = GenNet('ACT2', inputs, opt_spec, self.action_spec, force_cont_action, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=memory_size, max_steps=max_steps, force_det_out=False); outputs = self.actout(inputs)
            self.actout.optimizer_weights = util.optimizer_build(self.actout.optimizer['act'], self.actin.trainable_variables + self.actout.trainable_variables)
            util.net_build(self.actout, self.initializer)

        if arch in ('AC','MU','VPN','MU2',):
            opt_spec = [{'name':'value', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps}]
            if value_cont:
                value_spec = [{'net_type':0, 'dtype':compute_dtype, 'dtype_out':compute_dtype, 'is_discrete':False, 'num_components':8, 'event_shape':(1,), 'step_shape':tf.TensorShape((1,1))}]
                self.value = GenNet('VN', inputs, opt_spec, value_spec, False, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=memory_size, max_steps=max_steps, force_det_out=False); outputs = self.value(inputs)
            else: self.value = ValueNet('VN', inputs, opt_spec, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, num_heads=4, memory_size=memory_size); outputs = self.value(inputs)
            self.value.optimizer_weights = util.optimizer_build(self.value.optimizer['value'], self.value.trainable_variables)
            util.net_build(self.value, self.initializer)

        if arch in ('TRANS','MU','VPN','SPR','MU2','MU3','MU4',):
            inputs['actions'] = self.action_zero_out
            opt_spec = [{'name':'trans', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps}]
            latent_dist = 2; latent_spec = {'dtype':compute_dtype, 'num_latents':lat_batch_size, 'num_latents_trans':lat_batch_size_trans, 'event_shape':(latent_size,), 'num_components':8}
            self.trans = TransNet('TN', inputs, opt_spec, self.action_spec, latent_spec, latent_dist, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=memory_size_trans); outputs = self.trans(inputs)
            self.trans.optimizer_weights = util.optimizer_build(self.trans.optimizer['trans'], self.trans.trainable_variables)
            util.net_build(self.trans, self.initializer)
        if arch in ('MU','MU2','MU3','MU4',):
            opt_spec = [{'name':'rwd', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps}]
            reward_spec = [{'net_type':0, 'dtype':tf.float64, 'dtype_out':compute_dtype, 'is_discrete':False, 'num_components':16, 'event_shape':(1,), 'step_shape':tf.TensorShape((1,1))}]
            self.rwd = GenNet('RW', inputs, opt_spec, reward_spec, False, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=memory_size, max_steps=max_steps, force_det_out=False); outputs = self.rwd(inputs)
            self.rwd.optimizer_weights = util.optimizer_build(self.rwd.optimizer['rwd'], self.rwd.trainable_variables)
            util.net_build(self.rwd, self.initializer)
            opt_spec = [{'name':'done', 'type':'ar', 'schedule_type':'', 'learn_rate':self.learn_rate, 'float_eps':self.float_eps}]
            done_spec = [{'net_type':0, 'dtype':tf.bool, 'dtype_out':tf.int32, 'is_discrete':True, 'num_components':2, 'event_shape':(1,), 'step_shape':tf.TensorShape((1,1))}]
            self.done = GenNet('DO', inputs, opt_spec, done_spec, False, latent_size, net_blocks=2, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=4, memory_size=memory_size, max_steps=max_steps, force_det_out=False); outputs = self.done(inputs)
            self.done.optimizer_weights = util.optimizer_build(self.done.optimizer['done'], self.done.trainable_variables)
            util.net_build(self.done, self.initializer)



        metrics_loss = OrderedDict()
        metrics_loss['1rewards*'] = {'-ma_act':np.float64, '-rewards_act_total+':np.float64, '-rewards_act_final=':np.float64}
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
        if arch == 'MU4':
            metrics_loss['1rewards3*'] = {'-ma_PGL':np.float64, '-rewards_PGL_total+':np.float64, '-rewards_PGL_final=':np.float64}
            metrics_loss['1rewards2*'] = {'-ma_PG':np.float64, '-rewards_PG_total+':np.float64, '-rewards_PG_final=':np.float64}
            # metrics_loss['1extra'] = {'returns_pred':np.float64}
            metrics_loss['1nets'] = {'loss_PG':np.float64}; metrics_loss['1netsS'] = {'-std_PG':np.float64}
            # metrics_loss['1nets8'] = {'loss_PGL':np.float64}; metrics_loss['1nets8S'] = {'-std_PGL':np.float64}
            # metrics_loss['1nets4'] = {'loss_act':np.float64}
            # metrics_loss['1nets6'] = {'loss_trans':np.float64}
            # metrics_loss['1nets6'] = {'loss_trans':np.float64, 'loss_trans_img':np.float64}
            # metrics_loss['1nets2'] = {'loss_rwd':np.float64, 'loss_done':np.float64}
            # metrics_loss['1nets5'] = {'loss_rwd_img':np.float64, 'loss_done_img':np.float64}
            # metrics_loss['1nets7'] = {'loss_trans_ret':np.float64, 'loss_trans_dyn':np.float64}
            # metrics_loss['1nets3'] = {'loss_rwd_ret':np.float64, 'loss_done_ret':np.float64}
            # metrics_loss['1nets5'] = {'loss_rwd_dyn':np.float64, 'loss_done_dyn':np.float64}
            # metrics_loss['1extra2'] = {'return_entropy':np.float64}
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

    def check_stop(self, *args):
        if keyboard.is_pressed('ctrl+alt+k'): return np.asarray(True, bool)
        return np.asarray(False, bool)

    # TODO use ZMQ for remote messaging, latent pooling
    def transact_latents(self, *args):
        return [np.asarray([0,1,2], np.float64), np.asarray([2,1,0], np.float64)]


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
        if returns_target is not None:
            returns_target = tf.squeeze(tf.cast(returns_target, self.compute_dtype), axis=-1)
            # returns = returns_target - returns # _lRt
            returns = returns - returns_target # _lRtn
            # returns = tf.abs(returns_target - returns) / returns_target # _lRtan
            # returns = tf.abs(returns_target - returns) # _lRta
        if values is not None: returns = returns - tf.squeeze(tf.cast(values, self.compute_dtype), axis=-1)
        loss = loss_lik * returns # / self.float_maxroot
        # if values is not None: loss = loss * (-values)
        # if values is not None: loss = loss - loss_lik * values
        # if values is not None: loss = loss - values
        # if values is not None: loss = loss * (1.0 + values)
        # if values is not None: loss = loss * values

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)))
        if isinfnan > 0: tf.print('NaN/Inf PG loss:', loss)
        return loss





    # def VPN_actor(self, inputs):
    #     print("tracing -> GeneralAI VPN_actor")
    #     loss = {}
    #     loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     metric_entropy = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     metric_returns_pred = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #     obs, actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
    #     for i in range(self.obs_spec_len): obs[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['event_shape'])
    #     for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
    #     rewards = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     dones = tf.TensorArray(tf.bool, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     returns = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #     step = tf.constant(0)
    #     # while step < self.max_steps and not inputs['dones'][-1][0]:
    #     while not inputs['dones'][-1][0]:
    #         inputs_step = {}
    #         for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])

    #         with tf.GradientTape(persistent=True) as tape_action:
    #             rep_logits = self.rep(inputs, step=step); rep_dist = self.rep.dist(rep_logits)
    #             inputs_step['obs'] = rep_dist.sample()

    #         # outputs_img = self.VPN_imagine(inputs_step)
    #         # loss_img = self.VPN_img_learner(outputs_img)
    #         # action_rnd = [None]*self.action_spec_len
    #         # for i in range(self.action_spec_len):
    #         #     action_rnd[i] = tf.random.uniform((self.action_spec[i]['step_shape']), minval=self.action_spec[i]['min'], maxval=self.action_spec[i]['max'], dtype=self.action_spec[i]['dtype_out'])

    #         with tape_action:
    #             action_logits = self.action(inputs_step)
    #             action_dist, action, action_dis = [None]*self.action_spec_len, [None]*self.action_spec_len, [None]*self.action_spec_len
    #             for i in range(self.action_spec_len):
    #                 action_dist[i] = self.action.dist[i](action_logits[i])
    #                 action[i] = action_dist[i].sample()
    #                 actions[i] = actions[i].write(step, action[i][-1])
    #                 action_dis[i] = util.discretize(action[i], self.action_spec[i], self.force_cont_action)

    #         np_in = tf.numpy_function(self.env_step, action_dis, self.gym_step_dtypes)
    #         for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
    #         inputs['obs'], inputs['rewards'], inputs['dones'] = np_in[:-2], np_in[-2], np_in[-1]

    #         entropy = tf.constant([0.0], dtype=self.compute_dtype)
    #         inputs_step['actions'] = action
    #         trans_logits = self.trans(inputs_step); trans_dist = self.trans.dist(trans_logits)
    #         inputs_step['obs'] = trans_dist.sample()
    #         # entropy += trans_dist.entropy()

    #         if self.value_cont:
    #             value_logits = self.value(inputs_step); value_dist = self.value.dist[0](value_logits[0])
    #             values = value_dist.sample()
    #             entropy += value_dist.entropy()
    #         else: values = self.value(inputs_step)

    #         returns_pred = inputs['rewards'] + values
    #         with tape_action:
    #             loss_action = self.loss_PG(action_dist, action, returns_pred, entropy)
    #         gradients = tape_action.gradient(loss_action, self.rep.trainable_variables + self.action.trainable_variables)
    #         self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.action.trainable_variables))

    #         rewards = rewards.write(step, inputs['rewards'][-1])
    #         dones = dones.write(step, inputs['dones'][-1])
    #         returns_updt = returns.stack()
    #         returns_updt = returns_updt + inputs['rewards'][-1]
    #         returns = returns.unstack(returns_updt)
    #         returns = returns.write(step, [self.float64_zero])

    #         loss_actions = loss_actions.write(step, loss_action)
    #         metric_entropy = metric_entropy.write(step, entropy)
    #         metric_returns_pred = metric_returns_pred.write(step, returns_pred[0])

    #         step += 1

    #     outputs = {}
    #     out_obs, out_actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
    #     for i in range(self.obs_spec_len): out_obs[i] = obs[i].stack()
    #     for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
    #     outputs['obs'], outputs['actions'], outputs['rewards'], outputs['dones'], outputs['returns'] = out_obs, out_actions, rewards.stack(), dones.stack(), returns.stack()

    #     loss['action'], loss['entropy'], loss['returns_pred'] = loss_actions.concat(), metric_entropy.concat(), metric_returns_pred.concat()
    #     return outputs, inputs, loss

    # def VPN_return_learner(self, inputs, training=True):
    #     print("tracing -> GeneralAI VPN_return_learner")
    #     loss = {}
    #     loss_returns = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #     for step in tf.range(tf.shape(inputs['dones'])[0]):
    #         inputs_step = {}

    #         obs = [None]*self.obs_spec_len
    #         for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
    #         inputs_step['obs'] = obs
    #         with tf.GradientTape(persistent=True) as tape_value:
    #             rep_logits = self.rep(inputs_step, step=step); rep_dist = self.rep.dist(rep_logits)
    #             inputs_step['obs'] = rep_dist.sample()

    #         action = [None]*self.action_spec_len
    #         for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
    #         inputs_step['actions'] = action
    #         with tape_value:
    #             trans_logits = self.trans(inputs_step); trans_dist = self.trans.dist(trans_logits)
    #             inputs_step['obs'] = trans_dist.sample()

    #         returns = inputs['returns'][step:step+1]
    #         with tape_value:
    #             if self.value_cont:
    #                 value_logits = self.value(inputs_step); value_dist = self.value.dist[0](value_logits[0])
    #                 loss_return = self.loss_likelihood(value_dist, returns)
    #             else:
    #                 values = self.value(inputs_step)
    #                 loss_return = self.loss_diff(values, returns)
    #         gradients = tape_value.gradient(loss_return, self.rep.trainable_variables + self.trans.trainable_variables + self.value.trainable_variables)
    #         self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.value.trainable_variables))

    #         loss_returns = loss_returns.write(step, loss_return)

    #     loss['return'] = loss_returns.concat()
    #     return loss

    # def VPN_run_episode(self, inputs, episode, training=True):
    #     print("tracing -> GeneralAI VPN_run_episode")
    #     while not inputs['dones'][-1][0]:
    #         self.reset_states(); outputs, inputs, loss_actor = self.VPN_actor(inputs)
    #         self.reset_states(); loss_return = self.VPN_return_learner(outputs)

    #         metrics = [episode, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
    #             tf.math.reduce_mean(loss_actor['returns_pred']),
    #             tf.math.reduce_mean(loss_actor['action']),
    #             tf.math.reduce_mean(loss_actor['entropy']),
    #             tf.math.reduce_mean(loss_return['return']),
    #         ]
    #         dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

    # def VPN(self):
    #     print("tracing -> GeneralAI VPN")
    #     for episode in tf.range(self.max_episodes):
    #         tf.autograph.experimental.set_loop_options(parallel_iterations=1)
    #         np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
    #         for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
    #         inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}
    #         self.VPN_run_episode(inputs, episode)





    # def SPR_learner(self, inputs, num_img_steps, training=True):
    #     print("tracing -> GeneralAI SPR_learner")
    #     loss = {}
    #     loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_next_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_next_actions_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #     steps = tf.shape(inputs['dones'])[0]
    #     for step in tf.range(steps):
    #         inputs_step = {}

    #         obs = [None]*self.obs_spec_len
    #         for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
    #         inputs_step['obs'] = obs
    #         with tf.GradientTape(persistent=True) as tape_action, tf.GradientTape(persistent=True) as tape_next_action:
    #             rep_logits = self.rep(inputs_step, step=step); rep_dist = self.rep.dist(rep_logits)
    #             inputs_step['obs'] = rep_dist.sample()

    #         action = [None]*self.action_spec_len
    #         for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
    #         returns = inputs['returns'][step:step+1]
    #         with tape_action:
    #             action_logits = self.action(inputs_step)
    #             action_dist = [None]*self.action_spec_len
    #             for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
    #             loss_action = self.loss_PG(action_dist, action, returns)
    #         gradients = tape_action.gradient(loss_action, self.rep.trainable_variables + self.action.trainable_variables)
    #         self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.action.trainable_variables))
    #         loss_actions = loss_actions.write(step, loss_action)


    #         inputs_step['actions'] = action
    #         with tape_next_action:
    #             trans_logits = self.trans(inputs_step); trans_dist = self.trans.dist(trans_logits)
    #             inputs_step['obs'] = trans_dist.sample()

    #         dones = inputs['dones'][step:step+1]

    #         if not dones[-1][0]:
    #             self.reset_states(use_img=True)

    #             next_action = [None]*self.action_spec_len
    #             for i in range(self.action_spec_len): next_action[i] = inputs['actions'][i][step+1:step+2]; next_action[i].set_shape(self.action_spec[i]['step_shape'])
    #             with tape_next_action:
    #                 next_action_logits = self.action(inputs_step, use_img=True)
    #                 next_action_dist = [None]*self.action_spec_len
    #                 for i in range(self.action_spec_len): next_action_dist[i] = self.action.dist[i](next_action_logits[i])
    #                 loss_next_action = self.loss_likelihood(next_action_dist, next_action)
    #             gradients = tape_next_action.gradient(loss_next_action, self.rep.trainable_variables + self.trans.trainable_variables + self.action.trainable_variables)
    #             self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.action.trainable_variables))
    #             loss_next_actions = loss_next_actions.write(step, loss_next_action)


    #             loss_next_action_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #             step_img, step_img_max = step+2, step+2+num_img_steps
    #             while step_img < step_img_max and step_img < steps:

    #                 inputs_step['actions'] = next_action
    #                 with tf.GradientTape(persistent=True) as tape_next_action_img:
    #                     trans_logits = self.trans(inputs_step, use_img=True); trans_dist = self.trans.dist(trans_logits)
    #                     inputs_step['obs'] = trans_dist.sample()

    #                 next_action = [None]*self.action_spec_len
    #                 for i in range(self.action_spec_len): next_action[i] = inputs['actions'][i][step_img:step_img+1]; next_action[i].set_shape(self.action_spec[i]['step_shape'])
    #                 with tape_next_action_img:
    #                     next_action_logits = self.action(inputs_step, use_img=True)
    #                     next_action_dist = [None]*self.action_spec_len
    #                     for i in range(self.action_spec_len): next_action_dist[i] = self.action.dist[i](next_action_logits[i])
    #                     loss_next_action = self.loss_likelihood(next_action_dist, next_action)
    #                 gradients = tape_next_action_img.gradient(loss_next_action, self.trans.trainable_variables + self.action.trainable_variables)
    #                 self.net.optimizer['act'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.action.trainable_variables))
    #                 loss_next_action_img_accu = loss_next_action_img_accu.write(step_img-2, loss_next_action)

    #                 step_img += 1
    #             loss_next_actions_img = loss_next_actions_img.write(step, tf.math.reduce_mean(loss_next_action_img_accu.stack(), axis=0))

    #     loss['action'], loss['next_action'], loss['next_action_img'] = loss_actions.concat(), loss_next_actions.concat(), loss_next_actions_img.concat()
    #     return loss

    # def SPR_run_episode(self, inputs, episode, training=True):
    #     print("tracing -> GeneralAI SPR_run_episode")
    #     while not inputs['dones'][-1][0]:
    #         self.reset_states(); outputs, inputs = self.PG_actor(inputs)
    #         self.reset_states(); loss = self.SPR_learner(outputs, num_img_steps=2)

    #         metrics = [episode, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
    #             tf.math.reduce_mean(loss['action']),
    #             tf.math.reduce_mean(loss['next_action']),
    #             tf.math.reduce_mean(loss['next_action_img']),
    #         ]
    #         dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

    # def SPR(self):
    #     print("tracing -> GeneralAI SPR")
    #     for episode in tf.range(self.max_episodes):
    #         tf.autograph.experimental.set_loop_options(parallel_iterations=1)
    #         np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
    #         for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
    #         inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}
    #         self.SPR_run_episode(inputs, episode)





    # def MU2_img_learner(self, inputs):
    #     print("tracing -> GeneralAI MU2_img_learner")
    #     loss = {}
    #     loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     # metric_entropy = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     # metric_returns_pred = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #     inputs_step, dones = {'obs':inputs['obs'], 'actions':self.action_zero_out}, tf.constant([[False]])
    #     step = tf.constant(0)
    #     while step < 4 and not dones[-1][0]:
    #     # while not dones[-1][0]:

    #         with tf.GradientTape(persistent=True) as tape_action:
    #             action_logits = self.action(inputs_step, use_img=True)
    #             action_dist, action = [None]*self.action_spec_len, [None]*self.action_spec_len
    #             for i in range(self.action_spec_len):
    #                 action_dist[i] = self.action.dist[i](action_logits[i])
    #                 action[i] = action_dist[i].sample()

    #         inputs_step['actions'] = action
    #         trans_logits = self.trans(inputs_step, use_img=True); trans_dist = self.trans.dist(trans_logits)
    #         inputs_step['obs'] = trans_dist.sample()
    #         # entropy = trans_dist.entropy()
    #         # entropy = tf.constant([0.0], dtype=self.compute_dtype)

    #         rwd_logits = self.rwd(inputs_step, use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #         done_logits = self.done(inputs_step, use_img=True); done_dist = self.done.dist[0](done_logits[0])
    #         rewards, dones = rwd_dist.sample(), tf.cast(done_dist.sample(), tf.bool)

    #         if self.value_cont:
    #             value_logits = self.value(inputs_step, use_img=True); value_dist = self.value.dist[0](value_logits[0])
    #             values = value_dist.sample()
    #             # entropy = entropy + value_dist.entropy()
    #         else: values = self.value(inputs_step, use_img=True)

    #         returns_pred = rewards + values
    #         with tape_action:
    #             # loss_action = self.loss_PG(action_dist, action, returns_pred, entropy)
    #             loss_action = self.loss_PG(action_dist, action, returns_pred)
    #         gradients = tape_action.gradient(loss_action, self.action.trainable_variables)
    #         self.net.optimizer['act'].apply_gradients(zip(gradients, self.action.trainable_variables))
    #         loss_actions = loss_actions.write(step, loss_action)
    #         # metric_entropy = metric_entropy.write(step, entropy)
    #         # metric_returns_pred = metric_returns_pred.write(step, returns_pred[0])

    #         step += 1

    #     # loss['action'], loss['entropy'], loss['returns_pred'] = loss_actions.stack(), metric_entropy.stack(), metric_returns_pred.stack()
    #     loss['action'] = loss_actions.stack()
    #     return loss

    # def MU2_actor(self, inputs):
    #     print("tracing -> GeneralAI MU2_actor")
    #     loss = {}
    #     loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     # metric_entropy = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     # metric_returns_pred = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #     obs, actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
    #     for i in range(self.obs_spec_len): obs[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['event_shape'])
    #     for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
    #     rewards = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     dones = tf.TensorArray(tf.bool, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     returns = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #     step = tf.constant(0)
    #     # while step < self.max_steps and not inputs['dones'][-1][0]:
    #     while not inputs['dones'][-1][0]:
    #         inputs_step = {}
    #         for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])

    #         rep_logits = self.rep(inputs, step=step); rep_dist = self.rep.dist(rep_logits)
    #         inputs_step['obs'] = rep_dist.sample()

    #         for img_traj in tf.range(4):
    #             self.reset_states(use_img=True); loss_img = self.MU2_img_learner(inputs_step)
    #             # action_rnd = [None]*self.action_spec_len
    #             # for i in range(self.action_spec_len):
    #             #     action_rnd[i] = tf.random.uniform((self.action_spec[i]['step_shape']), minval=self.action_spec[i]['min'], maxval=self.action_spec[i]['max'], dtype=self.action_spec[i]['dtype_out'])

    #             loss_actions = loss_actions.write(step, tf.math.reduce_mean(loss_img['action'], axis=0))
    #             # metric_entropy = metric_entropy.write(step, tf.math.reduce_mean(loss_img['entropy'], axis=0))
    #             # metric_returns_pred = metric_returns_pred.write(step, tf.math.reduce_mean(loss_img['returns_pred'], axis=0))

    #         action_logits = self.action(inputs_step)
    #         action_dist, action, action_dis = [None]*self.action_spec_len, [None]*self.action_spec_len, [None]*self.action_spec_len
    #         for i in range(self.action_spec_len):
    #             action_dist[i] = self.action.dist[i](action_logits[i])
    #             action[i] = action_dist[i].sample()
    #             actions[i] = actions[i].write(step, action[i][-1])
    #             action_dis[i] = util.discretize(action[i], self.action_spec[i], self.force_cont_action)

    #         np_in = tf.numpy_function(self.env_step, action_dis, self.gym_step_dtypes)
    #         for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
    #         inputs['obs'], inputs['rewards'], inputs['dones'] = np_in[:-2], np_in[-2], np_in[-1]

    #         inputs_step['actions'] = action
    #         trans_logits = self.trans(inputs_step); trans_dist = self.trans.dist(trans_logits)
    #         inputs_step['obs'] = trans_dist.sample()

    #         rwd_logits = self.rwd(inputs_step); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #         done_logits = self.done(inputs_step); done_dist = self.done.dist[0](done_logits[0])

    #         if self.value_cont: value_logits = self.value(inputs_step); value_dist = self.value.dist[0](value_logits[0])
    #         else: values = self.value(inputs_step)

    #         rewards = rewards.write(step, inputs['rewards'][-1])
    #         dones = dones.write(step, inputs['dones'][-1])
    #         returns_updt = returns.stack()
    #         returns_updt = returns_updt + inputs['rewards'][-1]
    #         returns = returns.unstack(returns_updt)
    #         returns = returns.write(step, [self.float64_zero])

    #         step += 1

    #     outputs = {}
    #     out_obs, out_actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
    #     for i in range(self.obs_spec_len): out_obs[i] = obs[i].stack()
    #     for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
    #     outputs['obs'], outputs['actions'], outputs['rewards'], outputs['dones'], outputs['returns'] = out_obs, out_actions, rewards.stack(), dones.stack(), returns.stack()

    #     # loss['action'], loss['entropy'], loss['returns_pred'] = loss_actions.concat(), metric_entropy.concat(), metric_returns_pred.concat()
    #     loss['action'] = loss_actions.concat()
    #     return outputs, inputs, loss

    # def MU2_learner(self, inputs, num_img_steps, training=True):
    #     print("tracing -> GeneralAI MU2_learner")
    #     loss = {}
    #     # loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_rewards = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_dones = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_returns = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_next_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #     loss_rewards_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_dones_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_returns_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_next_actions_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #     steps = tf.shape(inputs['dones'])[0]
    #     for step in tf.range(steps):
    #         inputs_step = {}

    #         obs = [None]*self.obs_spec_len
    #         for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
    #         inputs_step['obs'] = obs
    #         with tf.GradientTape(persistent=True) as tape_action, tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done, tf.GradientTape(persistent=True) as tape_value, tf.GradientTape(persistent=True) as tape_next_action:
    #             rep_logits = self.rep(inputs_step, step=step); rep_dist = self.rep.dist(rep_logits)
    #             inputs_step['obs'] = rep_dist.sample()

    #         action = [None]*self.action_spec_len
    #         for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
    #         # returns = inputs['returns'][step:step+1]
    #         # returns = inputs['rewards'][step:step+1] + returns
    #         # with tape_action:
    #         #     action_logits = self.action(inputs_step)
    #         #     action_dist = [None]*self.action_spec_len
    #         #     for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
    #         #     loss_action = self.loss_PG(action_dist, action, returns)
    #         # gradients = tape_action.gradient(loss_action, self.rep.trainable_variables + self.action.trainable_variables)
    #         # self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.action.trainable_variables))
    #         # loss_actions = loss_actions.write(step, loss_action)



    #         inputs_step['actions'] = action
    #         with tape_reward, tape_done, tape_value, tape_next_action:
    #             trans_logits = self.trans(inputs_step); trans_dist = self.trans.dist(trans_logits)
    #             inputs_step['obs'] = trans_dist.sample()

    #         rewards = inputs['rewards'][step:step+1]
    #         with tape_reward:
    #             rwd_logits = self.rwd(inputs_step); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #             loss_reward = self.loss_likelihood(rwd_dist, rewards)
    #         gradients = tape_reward.gradient(loss_reward, self.rep.trainable_variables + self.trans.trainable_variables + self.rwd.trainable_variables)
    #         self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.rwd.trainable_variables))
    #         loss_rewards = loss_rewards.write(step, loss_reward)

    #         dones = inputs['dones'][step:step+1]
    #         with tape_done:
    #             done_logits = self.done(inputs_step); done_dist = self.done.dist[0](done_logits[0])
    #             loss_done = self.loss_likelihood(done_dist, dones)
    #         gradients = tape_done.gradient(loss_done, self.rep.trainable_variables + self.trans.trainable_variables + self.done.trainable_variables)
    #         self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.done.trainable_variables))
    #         loss_dones = loss_dones.write(step, loss_done)

    #         if not dones[-1][0]:
    #             returns = inputs['returns'][step:step+1]
    #             with tape_value:
    #                 if self.value_cont:
    #                     value_logits = self.value(inputs_step); value_dist = self.value.dist[0](value_logits[0])
    #                     loss_return = self.loss_likelihood(value_dist, returns)
    #                 else:
    #                     values = self.value(inputs_step)
    #                     loss_return = self.loss_diff(values, returns)
    #             gradients = tape_value.gradient(loss_return, self.rep.trainable_variables + self.trans.trainable_variables + self.value.trainable_variables)
    #             self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.value.trainable_variables))
    #             loss_returns = loss_returns.write(step, loss_return)

    #             self.reset_states(use_img=True)

    #             next_action = [None]*self.action_spec_len
    #             for i in range(self.action_spec_len): next_action[i] = inputs['actions'][i][step+1:step+2]; next_action[i].set_shape(self.action_spec[i]['step_shape'])
    #             with tape_next_action:
    #                 next_action_logits = self.action(inputs_step, use_img=True)
    #                 next_action_dist = [None]*self.action_spec_len
    #                 for i in range(self.action_spec_len): next_action_dist[i] = self.action.dist[i](next_action_logits[i])
    #                 loss_next_action = self.loss_likelihood(next_action_dist, next_action)
    #             gradients = tape_next_action.gradient(loss_next_action, self.rep.trainable_variables + self.trans.trainable_variables + self.action.trainable_variables)
    #             self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.action.trainable_variables))
    #             loss_next_actions = loss_next_actions.write(step, loss_next_action)



    #             loss_reward_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #             loss_done_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #             loss_return_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #             loss_next_action_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #             step_img, step_img_max = step+1, step+1+num_img_steps
    #             while step_img < step_img_max and step_img < steps:

    #                 inputs_step['actions'] = next_action
    #                 with tf.GradientTape(persistent=True) as tape_reward_img, tf.GradientTape(persistent=True) as tape_done_img, tf.GradientTape(persistent=True) as tape_value_img, tf.GradientTape(persistent=True) as tape_next_action_img:
    #                     trans_logits = self.trans(inputs_step, use_img=True); trans_dist = self.trans.dist(trans_logits)
    #                     inputs_step['obs'] = trans_dist.sample()

    #                 rewards = inputs['rewards'][step_img:step_img+1]
    #                 with tape_reward_img:
    #                     rwd_logits = self.rwd(inputs_step, use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #                     loss_reward = self.loss_likelihood(rwd_dist, rewards)
    #                 gradients = tape_reward_img.gradient(loss_reward, self.trans.trainable_variables + self.rwd.trainable_variables)
    #                 self.net.optimizer['act'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.rwd.trainable_variables))
    #                 loss_reward_img_accu = loss_reward_img_accu.write(step_img-1, loss_reward)

    #                 dones = inputs['dones'][step_img:step_img+1]
    #                 with tape_done_img:
    #                     done_logits = self.done(inputs_step, use_img=True); done_dist = self.done.dist[0](done_logits[0])
    #                     loss_done = self.loss_likelihood(done_dist, dones)
    #                 gradients = tape_done_img.gradient(loss_done, self.trans.trainable_variables + self.done.trainable_variables)
    #                 self.net.optimizer['act'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.done.trainable_variables))
    #                 loss_done_img_accu = loss_done_img_accu.write(step_img-1, loss_done)

    #                 step_img += 1
    #                 if step_img < steps:
    #                     returns = inputs['returns'][step_img-1:step_img]
    #                     with tape_value_img:
    #                         if self.value_cont:
    #                             value_logits = self.value(inputs_step, use_img=True); value_dist = self.value.dist[0](value_logits[0])
    #                             loss_return = self.loss_likelihood(value_dist, returns)
    #                         else:
    #                             values = self.value(inputs_step, use_img=True)
    #                             loss_return = self.loss_diff(values, returns)
    #                     gradients = tape_value_img.gradient(loss_return, self.trans.trainable_variables + self.value.trainable_variables)
    #                     self.net.optimizer['act'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.value.trainable_variables))
    #                     loss_return_img_accu = loss_return_img_accu.write(step_img-2, loss_return)

    #                     next_action = [None]*self.action_spec_len
    #                     for i in range(self.action_spec_len): next_action[i] = inputs['actions'][i][step_img:step_img+1]; next_action[i].set_shape(self.action_spec[i]['step_shape'])
    #                     with tape_next_action_img:
    #                         next_action_logits = self.action(inputs_step, use_img=True)
    #                         next_action_dist = [None]*self.action_spec_len
    #                         for i in range(self.action_spec_len): next_action_dist[i] = self.action.dist[i](next_action_logits[i])
    #                         loss_next_action = self.loss_likelihood(next_action_dist, next_action)
    #                     gradients = tape_next_action_img.gradient(loss_next_action, self.trans.trainable_variables + self.action.trainable_variables)
    #                     self.net.optimizer['act'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.action.trainable_variables))
    #                     loss_next_action_img_accu = loss_next_action_img_accu.write(step_img-2, loss_next_action)
    #             loss_rewards_img = loss_rewards_img.write(step, tf.math.reduce_mean(loss_reward_img_accu.stack(), axis=0))
    #             loss_dones_img = loss_dones_img.write(step, tf.math.reduce_mean(loss_done_img_accu.stack(), axis=0))
    #             loss_returns_img = loss_returns_img.write(step, tf.math.reduce_mean(loss_return_img_accu.stack(), axis=0))
    #             loss_next_actions_img = loss_next_actions_img.write(step, tf.math.reduce_mean(loss_next_action_img_accu.stack(), axis=0))


    #     # loss['action'] = loss_actions.concat()
    #     loss['reward'], loss['done'], loss['return'], loss['next_action'] = loss_rewards.concat(), loss_dones.concat(), loss_returns.concat(), loss_next_actions.concat()
    #     loss['reward_img'], loss['done_img'], loss['return_img'], loss['next_action_img'] = loss_rewards_img.concat(), loss_dones_img.concat(), loss_returns_img.concat(), loss_next_actions_img.concat()
    #     return loss

    # def MU2_run_episode(self, inputs, episode, training=True):
    #     print("tracing -> GeneralAI MU2_run_episode")
    #     while not inputs['dones'][-1][0]:
    #         self.reset_states(); outputs, inputs, loss_actor = self.MU2_actor(inputs)
    #         self.reset_states(); loss = self.MU2_learner(outputs, num_img_steps=4)

    #         metrics = [episode, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
    #             tf.math.reduce_mean(loss_actor['action']),
    #             # tf.math.reduce_mean(loss['action']),
    #             tf.math.reduce_mean(loss['reward']), tf.math.reduce_mean(loss['reward_img']),
    #             tf.math.reduce_mean(loss['done']), tf.math.reduce_mean(loss['done_img']),
    #             tf.math.reduce_mean(loss['return']), tf.math.reduce_mean(loss['return_img']),
    #             tf.math.reduce_mean(loss['next_action']), tf.math.reduce_mean(loss['next_action_img']),
    #             # tf.math.reduce_mean(loss['reward']),
    #             # tf.math.reduce_mean(loss['done']),
    #             # tf.math.reduce_mean(loss['return']),
    #             # tf.math.reduce_mean(loss['next_action']),
    #         ]
    #         dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

    # def MU2(self):
    #     print("tracing -> GeneralAI MU2")
    #     for episode in tf.range(self.max_episodes):
    #         tf.autograph.experimental.set_loop_options(parallel_iterations=1)
    #         np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
    #         for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
    #         inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}
    #         self.MU2_run_episode(inputs, episode)





    # def MU3_img_actor(self, inputs):
    #     print("tracing -> GeneralAI MU3_img_actor")
    #     obs = tf.TensorArray(self.latent_spec['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])
    #     # actions = [None]*self.action_spec_len
    #     # for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
    #     rewards = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     returns = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     metric_entropy_rwd = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     metric_entropy_done = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #     inputs_step = {'obs':inputs['obs'], 'actions':inputs['actions']}
    #     dones = tf.constant([[False]])
    #     # action_first, values, entropy = self.action_zero_out, tf.constant([[0.0]], dtype=self.compute_dtype), tf.constant(0.0, dtype=self.compute_dtype)

    #     step = tf.constant(0)
    #     # while step < 4 and not dones[-1][0]:
    #     while not dones[-1][0]:
    #         obs = obs.write(step, inputs_step['obs'])
    #         trans_logits = self.trans(inputs_step, use_img=True); trans_dist = self.trans.dist(trans_logits)
    #         inputs_step['obs'] = trans_dist.sample()

    #         rwd_logits = self.rwd(inputs_step, use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #         done_logits = self.done(inputs_step, use_img=True); done_dist = self.done.dist[0](done_logits[0])
    #         reward, dones = tf.cast(rwd_dist.sample(), tf.float64), tf.cast(done_dist.sample(), tf.bool)
    #         entropy_rwd, entropy_done = rwd_dist.entropy(), done_dist.entropy()
    #         metric_entropy_rwd = metric_entropy_rwd.write(step, entropy_rwd)
    #         metric_entropy_done = metric_entropy_done.write(step, entropy_done)

    #         # if self.value_cont:
    #         #     value_logits = self.value(inputs_step, use_img=True); value_dist = self.value.dist[0](value_logits[0])
    #         #     values = value_dist.sample()
    #         # else: values = self.value(inputs_step, use_img=True)

    #         rewards = rewards.write(step, reward[-1])
    #         returns_updt = returns.stack()
    #         returns_updt = returns_updt + reward[-1]
    #         returns = returns.unstack(returns_updt)
    #         returns = returns.write(step, [self.float64_zero])

    #         action = self.action_zero_out
    #         if not dones[-1][0]:
    #             action_logits = self.action(inputs_step, use_img=True)
    #             action_dist, action = [None]*self.action_spec_len, [None]*self.action_spec_len
    #             for i in range(self.action_spec_len):
    #                 # logits_rnd = tf.random.uniform(tf.shape(action_logits[i]), minval=-0.1, maxval=0.1, dtype=self.compute_dtype)
    #                 # logits_rnd += action_logits[i] * entropy
    #                 action_dist[i] = self.action.dist[i](action_logits[i])
    #                 action[i] = action_dist[i].sample()
    #                 # actions[i] = actions[i].write(step, action[i][-1])
    #             # if step == 0: action_first = action
    #         inputs_step['actions'] = action

    #         step += 1

    #     # returns = returns.write(step, [self.float64_zero])
    #     # returns_updt = returns.stack()
    #     # returns_updt = returns_updt + values[-1]
    #     # returns = returns.unstack(returns_updt)
    #     # returns_first = returns.stack()[0] + values[-1]
    #     # returns_first = returns.stack()[:1]

    #     outputs = {}
    #     # out_actions = [None]*self.action_spec_len
    #     # for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
    #     # outputs['actions'], outputs['returns'] = out_actions, returns.stack()
    #     # outputs['actions'], outputs['returns'] = action_first, returns_first
    #     # outputs['returns'] = returns_first
    #     outputs['obs'], outputs['rewards'], outputs['returns'] = obs.stack(), rewards.stack(), returns.stack()
    #     outputs['entropy_rwd'], outputs['entropy_done'] = tf.math.reduce_mean(metric_entropy_rwd.stack(), axis=0), tf.math.reduce_mean(metric_entropy_done.stack(), axis=0)
    #     return outputs

    # def MU3_actor(self, inputs):
    #     print("tracing -> GeneralAI MU3_actor")
    #     loss = {}
    #     loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_rewards = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_dones = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     metric_entropy = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     metric_returns_pred = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #     obs, actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
    #     for i in range(self.obs_spec_len): obs[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['event_shape'])
    #     for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
    #     rewards = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     dones = tf.TensorArray(tf.bool, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     returns = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #     step = tf.constant(0)
    #     # while step < self.max_steps and not inputs['dones'][-1][0]:
    #     while not inputs['dones'][-1][0]:
    #         inputs_step = {'obs':self.latent_zero, 'actions':self.action_zero_out}
    #         for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])

    #         with tf.GradientTape(persistent=True) as tape_action, tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done:
    #             rep_logits = self.rep(inputs, step=step); rep_dist = self.rep.dist(rep_logits)
    #             # if step != 0:
    #             inputs_step['obs'] = rep_dist.sample()

    #         # action, returns_max = self.action_zero_out, self.float_min
    #         # for img_traj in tf.range(4):
    #         #     self.reset_states(use_img=True); outputs_img = self.MU3_img_actor(inputs_step)
    #         #     if outputs_img['returns'] > returns_max:
    #         #         returns_max = outputs_img['returns']
    #         #         action = outputs_img['actions']
    #         # with tape_action:
    #         #     action_logits = self.action(inputs_step)
    #         #     action_dist = [None]*self.action_spec_len
    #         #     for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])


    #         # TODO can I add ARS/MCTS here?
    #         action = [None]*self.action_spec_len
    #         for i in range(self.action_spec_len):
    #             action[i] = tf.random.uniform((self.action_spec[i]['step_shape']), minval=self.action_spec[i]['min'], maxval=self.action_spec[i]['max'], dtype=self.action_spec[i]['dtype_out'])


    #         # # with tape_action:
    #         # action_logits = self.action(inputs_step)
    #         # action_dist, action = [None]*self.action_spec_len, [None]*self.action_spec_len
    #         # for i in range(self.action_spec_len):
    #         #     action_dist[i] = self.action.dist[i](action_logits[i])
    #         #     action[i] = action_dist[i].sample()
    #         # self.action.reset_states(use_img=True)
    #         # inputs_step['actions'] = action
    #         # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
    #         # outputs_img = self.MU3_img_actor(inputs_step)

    #         # action_logits = self.action(inputs_step, store_memory=False, use_img=True)
    #         # # action_dist, action = [None]*self.action_spec_len, [None]*self.action_spec_len
    #         # for i in range(self.action_spec_len):
    #         #     action_dist[i] = self.action.dist[i](action_logits[i])
    #         #     # action[i] = action_dist[i].sample(); action[i].set_shape(self.action_spec[i]['step_shape'])
    #         #     action[i] = action_dist[i].sample()
    #         # inputs_step['actions'] = action
    #         # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
    #         # outputs_img = self.MU3_img_actor(inputs_step)

    #         # action_logits = self.action(inputs_step, store_memory=False, use_img=True)
    #         # # action_dist, action = [None]*self.action_spec_len, [None]*self.action_spec_len
    #         # for i in range(self.action_spec_len):
    #         #     action_dist[i] = self.action.dist[i](action_logits[i])
    #         #     # action[i] = action_dist[i].sample(); action[i].set_shape(self.action_spec[i]['step_shape'])
    #         #     action[i] = action_dist[i].sample()
    #         # inputs_step['actions'] = action
    #         # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
    #         # outputs_img = self.MU3_img_actor(inputs_step)

    #         # with tape_action:
    #         #     action_logits = self.action(inputs_step, store_memory=False, use_img=True)
    #         #     # action_dist, action = [None]*self.action_spec_len, [None]*self.action_spec_len
    #         #     for i in range(self.action_spec_len):
    #         #         action_dist[i] = self.action.dist[i](action_logits[i])
    #         #         # action[i] = action_dist[i].sample(); action[i].set_shape(self.action_spec[i]['step_shape'])
    #         #         action[i] = action_dist[i].sample()
    #         # inputs_step['actions'] = action
    #         # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
    #         # outputs_img = self.MU3_img_actor(inputs_step)


    #         action_dis = [None]*self.action_spec_len
    #         for i in range(self.action_spec_len):
    #             actions[i] = actions[i].write(step, action[i][-1])
    #             action_dis[i] = util.discretize(action[i], self.action_spec[i], self.force_cont_action)

    #         np_in = tf.numpy_function(self.env_step, action_dis, self.gym_step_dtypes)
    #         for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
    #         inputs['obs'], inputs['rewards'], inputs['dones'] = np_in[:-2], np_in[-2], np_in[-1]

    #         # entropy = tf.constant([0.0], dtype=self.compute_dtype)
    #         # inputs_step['actions'] = action
    #         with tape_reward, tape_done:
    #             trans_logits = self.trans(inputs_step); trans_dist = self.trans.dist(trans_logits)
    #             inputs_step['obs'] = trans_dist.sample()
    #         # entropy += trans_dist.entropy()

    #         # if self.value_cont:
    #         #     value_logits = self.value(inputs_step); value_dist = self.value.dist[0](value_logits[0])
    #         #     values = value_dist.sample()
    #         #     entropy += value_dist.entropy()
    #         # else: values = self.value(inputs_step)

    #         # if action[0] == tf.cast(inputs['obs'][0], dtype=tf.int32):
    #         #     tf.print('test')

    #         # values, entropy = outputs_img['returns'][:1], outputs_img['entropy_rwd']
    #         # returns_pred = inputs['rewards'] + values
    #         # # returns_pred = inputs['rewards']
    #         # with tape_action:
    #         #     # loss_action = self.loss_PG(action_dist, action, returns_pred, entropy)
    #         #     loss_action = self.loss_PG(action_dist, action, returns_pred)
    #         #     # loss_action = self.loss_likelihood(action_dist, action)
    #         # gradients = tape_action.gradient(loss_action, self.rep.trainable_variables + self.action.trainable_variables)
    #         # self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.action.trainable_variables))
    #         # loss_actions = loss_actions.write(step, loss_action)
    #         # metric_entropy = metric_entropy.write(step, entropy)
    #         # metric_returns_pred = metric_returns_pred.write(step, returns_pred[0])


    #         with tape_reward:
    #             rwd_logits = self.rwd(inputs_step); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #             loss_reward = self.loss_likelihood(rwd_dist, inputs['rewards'])
    #         gradients = tape_reward.gradient(loss_reward, self.rep.trainable_variables + self.trans.trainable_variables + self.rwd.trainable_variables)
    #         self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.rwd.trainable_variables))
    #         loss_rewards = loss_rewards.write(step, loss_reward)

    #         with tape_done:
    #             done_logits = self.done(inputs_step); done_dist = self.done.dist[0](done_logits[0])
    #             loss_done = self.loss_likelihood(done_dist, inputs['dones'])
    #         gradients = tape_done.gradient(loss_done, self.rep.trainable_variables + self.trans.trainable_variables + self.done.trainable_variables)
    #         self.net.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.done.trainable_variables))
    #         loss_dones = loss_dones.write(step, loss_done)


    #         rewards = rewards.write(step, inputs['rewards'][-1])
    #         dones = dones.write(step, inputs['dones'][-1])
    #         returns_updt = returns.stack()
    #         returns_updt = returns_updt + inputs['rewards'][-1]
    #         returns = returns.unstack(returns_updt)
    #         returns = returns.write(step, [self.float64_zero])

    #         step += 1

    #     outputs = {}
    #     out_obs, out_actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
    #     for i in range(self.obs_spec_len): out_obs[i] = obs[i].stack()
    #     for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
    #     outputs['obs'], outputs['actions'], outputs['rewards'], outputs['dones'], outputs['returns'] = out_obs, out_actions, rewards.stack(), dones.stack(), returns.stack()

    #     loss['action'], loss['reward'], loss['done'], loss['entropy'], loss['returns_pred'] = loss_actions.concat(), loss_rewards.concat(), loss_dones.concat(), metric_entropy.concat(), metric_returns_pred.concat()
    #     return outputs, inputs, loss

    # def MU3_dyn_learner(self, inputs, training=True):
    #     print("tracing -> GeneralAI MU3_dyn_learner")
    #     loss = {}
    #     loss_rewards = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #     loss_dones = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #     obs = [None]*self.obs_spec_len
    #     for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][:1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
    #     inputs_step = {'obs':obs, 'actions':self.action_zero_out}
    #     rep_logits = self.rep(inputs_step, step=0); rep_dist = self.rep.dist(rep_logits)
    #     inputs_step['obs'] = rep_dist.sample()

    #     for step in tf.range(tf.shape(inputs['dones'])[0]):
    #         action = [None]*self.action_spec_len
    #         for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
    #         inputs_step['actions'] = action
    #         with tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done:
    #             trans_logits = self.trans(inputs_step); trans_dist = self.trans.dist(trans_logits)
    #             inputs_step['obs'] = trans_dist.sample()

    #         with tape_reward:
    #             rwd_logits = self.rwd(inputs_step); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #             loss_reward = self.loss_likelihood(rwd_dist, inputs['rewards'][step:step+1])
    #         gradients = tape_reward.gradient(loss_reward, self.trans.trainable_variables)
    #         self.net.optimizer['act'].apply_gradients(zip(gradients, self.trans.trainable_variables))
    #         loss_rewards = loss_rewards.write(step, loss_reward)

    #         with tape_done:
    #             done_logits = self.done(inputs_step); done_dist = self.done.dist[0](done_logits[0])
    #             loss_done = self.loss_likelihood(done_dist, inputs['dones'][step:step+1])
    #         gradients = tape_done.gradient(loss_done, self.trans.trainable_variables)
    #         self.net.optimizer['act'].apply_gradients(zip(gradients, self.trans.trainable_variables))
    #         loss_dones = loss_dones.write(step, loss_done)

    #     loss['reward'], loss['done'] = loss_rewards.concat(), loss_dones.concat()
    #     return loss

    # def MU3_run_episode(self, inputs, episode, training=True):
    #     print("tracing -> GeneralAI MU3_run_episode")
    #     while not inputs['dones'][-1][0]:
    #         self.reset_states(); outputs, inputs, loss_actor = self.MU3_actor(inputs)
    #         self.reset_states(); loss_dyn = self.MU3_dyn_learner(outputs)

    #         metrics = [episode, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
    #             tf.math.reduce_mean(loss_actor['returns_pred']),
    #             tf.math.reduce_mean(loss_actor['action']),
    #             tf.math.reduce_mean(loss_actor['reward']), tf.math.reduce_mean(loss_actor['done']),
    #             tf.math.reduce_mean(loss_dyn['reward']), tf.math.reduce_mean(loss_dyn['done']),
    #             tf.math.reduce_mean(loss_actor['entropy']),
    #         ]
    #         # TODO add simulation episode time
    #         if self.trader: metrics += [tf.math.reduce_mean(tf.concat([outputs['obs'][3],inputs['obs'][3]],0)), inputs['obs'][3][-1][0],
    #             tf.math.reduce_mean(tf.concat([outputs['obs'][4],inputs['obs'][4]],0)), tf.math.reduce_mean(tf.concat([outputs['obs'][5],inputs['obs'][5]],0)),
    #             inputs['obs'][0][-1][0] - outputs['obs'][0][0][0],]
    #         dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

    # def MU3(self):
    #     print("tracing -> GeneralAI MU3")
    #     for episode in tf.range(self.max_episodes):
    #         tf.autograph.experimental.set_loop_options(parallel_iterations=1)
    #         np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
    #         for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
    #         inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}
    #         self.MU3_run_episode(inputs, episode)





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

    def gen_act(self, inputs, return_goal, use_img=False, store_real=False):
        inputs_act = {'obs':inputs['obs'], 'actions':[return_goal], 'step_size':inputs['step_size']}
        actin_logits = self.actin(inputs_act, use_img=use_img, store_real=store_real); actin_dist = self.actin.dist(actin_logits)
        inputs_act['obs'] = actin_dist.sample()
        action_logits = self.actout(inputs_act, use_img=use_img, store_real=store_real)
        action = [None]*self.action_spec_len
        for i in range(self.action_spec_len):
            action_dist = self.actout.dist[i](action_logits[i])
            action[i] = action_dist.sample()
        return action

    def MU4_img(self, inputs, gen, return_goal):
        print("tracing -> GeneralAI MU4_img")
        obs = tf.TensorArray(self.latent_spec['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])
        actions = [None]*self.action_spec_len
        for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
        # rewards = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))
        returns = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))
        step_sizes = tf.TensorArray(tf.int32, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))
        entropies = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        step_size, step_scale = 1, 0
        step_loc = self.attn_img_step_locs[step_scale]

        inputs_step = {'obs':inputs['obs'], 'actions':self.action_zero_out, 'step_size':step_size}
        step, dones = tf.constant(0), tf.constant([[False]])
        while not dones[-1][0]:
            obs = obs.write(step, inputs_step['obs'])
            step_sizes = step_sizes.write(step, [inputs_step['step_size']])

            action = self.action_zero_out
            if gen == 0: action = self.gen_PG(inputs_step, use_img=True)
            if gen == 1: action = self.gen_act(inputs_step, return_goal, use_img=True)
            if gen == 2: action = self.gen_rnd()
            for i in range(self.action_spec_len): actions[i] = actions[i].write(step, action[i][-1])
            inputs_step['actions'] = action

            if step == step_loc:
                step_size = self.attn_img_step_sizesT[step_scale]
                step_scale += 1
                if step_scale != self.attn_img_scales: step_loc = self.attn_img_step_locs[step_scale]
                # if step != step_loc: step_size = 1 # TODO remove
            inputs_step['step_size'] = step_size

            trans_logits = self.trans(inputs_step, use_img=True); trans_dist = self.trans.dist(trans_logits)
            inputs_step['obs'] = trans_dist.sample()

            rwd_logits = self.rwd(inputs_step, use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
            done_logits = self.done(inputs_step, use_img=True); done_dist = self.done.dist[0](done_logits[0])
            reward, dones = tf.cast(rwd_dist.sample(), tf.float64), tf.cast(done_dist.sample(), tf.bool)
            if step_scale == self.attn_img_scales: dones = tf.constant([[True]])

            entropy_rwd, entropy_done = rwd_dist.entropy(), done_dist.entropy()
            entropy = (entropy_rwd + entropy_done) * 0.1
            entropies = entropies.write(step, entropy)

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

    #     self.actin.reset_states(use_img=True); self.actout.reset_states(use_img=True)
    #     for step in tf.range(tf.shape(inputs['obs'])[0]):
    #         inputs_act = {'obs':inputs['obs'][step:step+1][0], 'actions':[inputs['returns'][step:step+1]], 'step_size':1}
    #         actin_logits = self.actin(inputs_act, use_img=True); actin_dist = self.actin.dist(actin_logits)
    #         inputs_act['obs'] = actin_dist.sample()
    #         action_logits = self.actout(inputs_act, use_img=True)

    #     inputs_act = {'obs':inputs['obs'][:1][0], 'actions':[inputs['returns'][:1]], 'step_size':1}
    #     with tf.GradientTape() as tape_action:
    #         actin_logits = self.actin(inputs_act, store_memory=False, use_img=True); actin_dist = self.actin.dist(actin_logits)
    #         inputs_act['obs'] = actin_dist.sample()
    #         action_logits = self.actout(inputs_act, store_memory=False, use_img=True)
    #         action_dist = [None]*self.action_spec_len
    #         for i in range(self.action_spec_len): action_dist[i] = self.actout.dist[i](action_logits[i])
    #         loss_action = self.loss_likelihood(action_dist, action)
    #     gradients = tape_action.gradient(loss_action, self.actin.trainable_variables + self.actout.trainable_variables) # self.rep.trainable_variables
    #     self.actout.optimizer['act'].apply_gradients(zip(gradients, self.actin.trainable_variables + self.actout.trainable_variables)) # self.rep.trainable_variables
    #     # loss_actions = loss_actions.write(step, loss_action)

    def MU4_img_learner(self, inputs, gen, training=True):
        print("tracing -> GeneralAI MU4_img_learner")
        loss = {}
        loss_PG = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_act = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        for step in tf.range(tf.shape(inputs['returns'])[0]):
            inputs_step = {'obs':inputs['obs'][step:step+1][0]}
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
            return_step = inputs['returns'][step:step+1]

            if gen == 0:
                with tf.GradientTape() as tape_PG:
                    action_logits = self.action(inputs_step, use_img=True)
                    action_dist = [None]*self.action_spec_len
                    for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
                    loss_action = self.loss_PG(action_dist, action, return_step)
                gradients = tape_PG.gradient(loss_action, self.action.trainable_variables)
                self.action.optimizer['action'].apply_gradients(zip(gradients, self.action.trainable_variables))
                loss_PG = loss_PG.write(step, loss_action)

            inputs_act = {'obs':inputs_step['obs'], 'actions':return_step, 'step_size':inputs['step_size'][step:step+1]}
            with tf.GradientTape() as tape_act:
                actin_logits = self.actin(inputs_act, use_img=True); actin_dist = self.actin.dist(actin_logits)
                inputs_act['obs'] = actin_dist.sample()
                action_logits = self.actout(inputs_act, use_img=True)
                action_dist = [None]*self.action_spec_len
                for i in range(self.action_spec_len): action_dist[i] = self.actout.dist[i](action_logits[i])
                loss_action = self.loss_likelihood(action_dist, action)
                # loss_action = self.loss_PG(action_dist, action, inputs['entropy'][step:step+1])
            gradients = tape_act.gradient(loss_action, self.actin.trainable_variables + self.actout.trainable_variables)
            self.actout.optimizer['act'].apply_gradients(zip(gradients, self.actin.trainable_variables + self.actout.trainable_variables))
            loss_act = loss_act.write(step, loss_action)

        loss['loss_PG'], loss['loss_act'] = loss_PG.concat(), loss_act.concat()
        return loss

    def train_trans(self, inputs, target_trans, target_reward, target_done, store_memory=True, training=True):
        inputs_alt = {'obs':inputs['obs'], 'actions':inputs['actions'], 'step_size':inputs['step_size']}
        with tf.GradientTape() as tape_trans, tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done:
            trans_logits = self.trans(inputs_alt, store_memory=store_memory); trans_dist = self.trans.dist(trans_logits)
            inputs_alt['obs'] = trans_dist.sample()
            loss_tran = self.loss_likelihood(trans_dist, target_trans)
            # loss_tran = self.loss_diff(inputs_alt['obs'], target_trans)
        gradients = tape_trans.gradient(loss_tran, self.trans.trainable_variables)
        self.trans.optimizer['trans'].apply_gradients(zip(gradients, self.trans.trainable_variables))

        with tape_reward:
            rwd_logits = self.rwd(inputs_alt, store_memory=store_memory); rwd_dist = self.rwd.dist[0](rwd_logits[0])
            loss_reward = self.loss_likelihood(rwd_dist, target_reward)
        gradients = tape_reward.gradient(loss_reward, self.rwd.trainable_variables) # self.trans.trainable_variables + self.rwd.trainable_variables
        self.rwd.optimizer['rwd'].apply_gradients(zip(gradients, self.rwd.trainable_variables)) # self.trans.trainable_variables + self.rwd.trainable_variables

        with tape_done:
            done_logits = self.done(inputs_alt, store_memory=store_memory); done_dist = self.done.dist[0](done_logits[0])
            loss_done = self.loss_likelihood(done_dist, target_done)
        gradients = tape_done.gradient(loss_done, self.done.trainable_variables) # self.trans.trainable_variables + self.done.trainable_variables
        self.done.optimizer['done'].apply_gradients(zip(gradients, self.done.trainable_variables)) # self.trans.trainable_variables + self.done.trainable_variables

        return loss_tran, loss_reward, loss_done

    def MU4_actor(self, inputs, gen, return_goal, return_goal_alt):
        print("tracing -> GeneralAI MU4_actor")
        loss = {}
        loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_trans = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_trans_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_rewards = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_dones = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_rewards_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_dones_img = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        # metric_entropy = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        # metric_returns_pred = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        obs, actions, returns = [None]*self.obs_spec_len, [None]*self.action_spec_len, [None]*self.attn_img_scales
        for i in range(self.obs_spec_len): obs[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['event_shape'])
        for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['event_shape'])
        for i in range(self.attn_img_scales): returns[i] = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))
        rewards = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))
        dones = tf.TensorArray(tf.bool, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        # obs_rep = tf.TensorArray(self.latent_spec['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])
        # obs_trans = tf.TensorArray(self.latent_spec['dtype'], size=0, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])
        # obs_trans_img = tf.TensorArray(self.latent_spec['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])


        dyn_img_obs = self.latent_zero
        step = tf.constant(0)
        while not inputs['dones'][-1][0]:
            for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])

            inputs_step = {'obs':self.latent_zero, 'actions':self.action_zero_out, 'step_size':1}
            # with tf.GradientTape(persistent=True) as tape_trans:
            # with tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done: # tf.GradientTape(persistent=True) as tape_action,
            # with tf.GradientTape(persistent=True) as tape_trans, tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done: # tf.GradientTape(persistent=True) as tape_action,
            rep_logits = self.rep(inputs, step=step); rep_dist = self.rep.dist(rep_logits)
            inputs_step['obs'] = rep_dist.sample()
            # obs_rep = obs_rep.write(step, inputs_step['obs'])

            # TODO change done to predict the number of steps until done
            # TODO train trans to match rep latent output (use cosine similarity if not using distribution latents)
            # TODO add transformer layer latents together for long horizon prediction and long history
            # TODO add ARS/MCTS/explorer here too (explorer is trained to find unique situations)
            # TODO train act on img but scale likelihood based on prediction entropy (higher entropy = flatter loss)
            # TODO scale likelihood based on action entropy (lower entropy = flatter loss)


            # TODO can I use linear layers in the transformer to compress (4 to 1, 64 to 1, full episode to 1, etc) the far (past+future) latents?

            # # ## _img
            # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
            # self.action.reset_states(use_img=True); self.actin.reset_states(use_img=True); self.actout.reset_states(use_img=True)
            # outputs_img = self.MU4_img(inputs_step, gen, return_goal)

            # self.action.reset_states(use_img=True); self.actin.reset_states(use_img=True); self.actout.reset_states(use_img=True)
            # loss_act = self.MU4_img_learner(outputs_img, gen)
            # loss_actions = loss_actions.write(step, tf.expand_dims(tf.math.reduce_mean(loss_act['loss_PG'], axis=0), axis=0))


            # ## _img2
            # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
            # outputs_img = self.MU4_img(inputs_step, 2, return_goal)
            # self.action.reset_states(use_img=True); self.actin.reset_states(use_img=True); self.actout.reset_states(use_img=True)
            # loss_act = self.MU4_img_learner(outputs_img, 2)

            # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
            # self.action.reset_states(use_img=True); self.actin.reset_states(use_img=True); self.actout.reset_states(use_img=True)
            # outputs_img = self.MU4_img(inputs_step, 0, return_goal)
            # self.action.reset_states(use_img=True); self.actin.reset_states(use_img=True); self.actout.reset_states(use_img=True)
            # loss_act = self.MU4_img_learner(outputs_img, 0)

            # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
            # self.action.reset_states(use_img=True); self.actin.reset_states(use_img=True); self.actout.reset_states(use_img=True)
            # outputs_img = self.MU4_img(inputs_step, 1, return_goal_alt)
            # self.action.reset_states(use_img=True); self.actin.reset_states(use_img=True); self.actout.reset_states(use_img=True)
            # loss_act = self.MU4_img_learner(outputs_img, 1)

            # self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
            # self.action.reset_states(use_img=True); self.actin.reset_states(use_img=True); self.actout.reset_states(use_img=True)
            # outputs_img = self.MU4_img(inputs_step, 1, return_goal)
            # self.action.reset_states(use_img=True); self.actin.reset_states(use_img=True); self.actout.reset_states(use_img=True)
            # loss_act = self.MU4_img_learner(outputs_img, 1)
            # loss_actions = loss_actions.write(step, tf.expand_dims(tf.math.reduce_mean(loss_act['loss_act'], axis=0), axis=0))


            action = self.action_zero_out
            if gen == 0:
                # action = self.gen_PG(inputs_step, use_img=True, store_real=True) # _img
                action = self.gen_PG(inputs_step)
            if gen == 1:
                # action = self.gen_act(inputs_step, return_goal, use_img=True, store_real=True) # _img
                action = self.gen_act(inputs_step, return_goal)
            if gen == 2:
                # action = self.gen_rnd()
                action = self.gen_PGL(inputs_step)



            action_dis = [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                actions[i] = actions[i].write(step, action[i][-1])
                action_dis[i] = util.discretize(action[i], self.action_spec[i], self.force_cont_action)

            np_in = tf.numpy_function(self.env_step, action_dis, self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs['obs'], inputs['rewards'], inputs['dones'] = np_in[:-2], np_in[-2], np_in[-1]
            return_goal -= inputs['rewards']
            return_goal_alt -= inputs['rewards']

            rewards = rewards.write(step, inputs['rewards'][-1])
            dones = dones.write(step, inputs['dones'][-1])

            for step_scale in range(self.attn_img_scales):
                returns[step_scale] = returns[step_scale].write(step, [self.float64_zero])
                returns_updt = returns[step_scale].stack()
                if step_scale != self.attn_img_scales-1:
                    step_size = self.attn_img_step_sizes[step_scale]
                    returns_temp = returns_updt[-step_size:] + inputs['rewards'][-1]
                    returns_updt = tf.concat([returns_updt[:-step_size], returns_temp], axis=0)
                else: returns_updt = returns_updt + inputs['rewards'][-1]
                returns[step_scale] = returns[step_scale].unstack(returns_updt)



            # ## _dyn2
            # if step >= 1:
            #     # inputs_step_img = {'obs':obs_trans.stack()[step-1:step][0], 'actions':action, 'step_size':1}
            #     inputs_step_img = {'obs':dyn_img_obs, 'actions':action, 'step_size':1}
            #     self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)

            #     ## _dyn7
            #     rep_logits = self.rep(inputs, step=step+1, store_memory=False); rep_dist = self.rep.dist(rep_logits)
            #     rep_target = rep_dist.sample()
            #     with tf.GradientTape() as tape_trans_img, tf.GradientTape(persistent=True) as tape_reward_img, tf.GradientTape(persistent=True) as tape_done_img:
            #         trans_logits = self.trans(inputs_step_img, use_img=True); trans_dist = self.trans.dist(trans_logits)
            #         inputs_step_img['obs'] = trans_dist.sample()
            #         loss_tran = self.loss_likelihood(trans_dist, rep_target)
            #         # loss_tran = self.loss_diff(inputs_step_img['obs'], rep_target)
            #     gradients = tape_trans_img.gradient(loss_tran, self.trans.trainable_variables)
            #     self.trans.optimizer['trans'].apply_gradients(zip(gradients, self.trans.trainable_variables))
            #     loss_trans_img = loss_trans_img.write(step, tf.expand_dims(tf.math.reduce_mean(loss_tran, axis=0), axis=0))

            #     # with tf.GradientTape(persistent=True) as tape_reward_img, tf.GradientTape(persistent=True) as tape_done_img:
            #     #     trans_logits = self.trans(inputs_step_img, use_img=True); trans_dist = self.trans.dist(trans_logits)
            #     #     inputs_step_img['obs'] = trans_dist.sample()
            #     # # obs_trans_img = obs_trans_img.write(step, inputs_step_img['obs'])

            #     with tape_reward_img:
            #         rwd_logits = self.rwd(inputs_step_img, use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
            #         loss_reward = self.loss_likelihood(rwd_dist, inputs['rewards'])
            #     gradients = tape_reward_img.gradient(loss_reward, self.rwd.trainable_variables) # self.trans.trainable_variables + self.rwd.trainable_variables
            #     self.rwd.optimizer['rwd'].apply_gradients(zip(gradients, self.rwd.trainable_variables)) # self.trans.trainable_variables + self.rwd.trainable_variables
            #     loss_rewards_img = loss_rewards_img.write(step, loss_reward)

            #     with tape_done_img:
            #         done_logits = self.done(inputs_step_img, use_img=True); done_dist = self.done.dist[0](done_logits[0])
            #         loss_done = self.loss_likelihood(done_dist, inputs['dones'])
            #     gradients = tape_done_img.gradient(loss_done, self.done.trainable_variables) # self.trans.trainable_variables + self.done.trainable_variables
            #     self.done.optimizer['done'].apply_gradients(zip(gradients, self.done.trainable_variables)) # self.trans.trainable_variables + self.done.trainable_variables
            #     loss_dones_img = loss_dones_img.write(step, loss_done)

            # ## _dyn4 # not the right history context: is real not imaginary, needs too much memory to store ongoing imagination trajectories from each old real state
            # inputs_step_img = {'obs':self.latent_zero, 'actions':action}
            # obs_img = obs_trans.stack()
            # for step_dyn in tf.range(tf.shape(obs_img)[0]):
            #     inputs_step_img['obs'] = obs_img[step_dyn]
            #     self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
            #     with tf.GradientTape(persistent=True) as tape_reward_img, tf.GradientTape(persistent=True) as tape_done_img:
            #         trans_logits = self.trans(inputs_step_img, use_img=True); trans_dist = self.trans.dist(trans_logits)
            #         inputs_step_img['obs'] = trans_dist.sample()
            #     obs_trans = obs_trans.write(step_dyn, inputs_step_img['obs'])

            #     with tape_reward_img:
            #         rwd_logits = self.rwd(inputs_step_img, use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
            #         loss_reward = self.loss_likelihood(rwd_dist, inputs['rewards'])
            #     gradients = tape_reward_img.gradient(loss_reward, self.trans.trainable_variables) # + self.rwd.trainable_variables
            #     self.rwd.optimizer['rwd'].apply_gradients(zip(gradients, self.trans.trainable_variables)) # + self.rwd.trainable_variables
            #     loss_rewards_img = loss_rewards_img.write(step, loss_reward)

            #     with tape_done_img:
            #         done_logits = self.done(inputs_step_img, use_img=True); done_dist = self.done.dist[0](done_logits[0])
            #         loss_done = self.loss_likelihood(done_dist, inputs['dones'])
            #     gradients = tape_done_img.gradient(loss_done, self.trans.trainable_variables) # + self.done.trainable_variables
            #     self.done.optimizer['done'].apply_gradients(zip(gradients, self.trans.trainable_variables)) # + self.done.trainable_variables
            #     loss_dones_img = loss_dones_img.write(step, loss_done)


            # ## _dyn6
            # rep_logits = self.rep(inputs, step=step+1, store_memory=False); rep_dist = self.rep.dist(rep_logits)
            # rep_target = rep_dist.sample()
            # inputs_step_dyn = {'obs':inputs_step['obs'], 'actions':action, 'step_size':1}
            # # with tf.GradientTape() as tape_trans:
            # with tape_trans:
            #     trans_logits = self.trans(inputs_step_dyn); trans_dist = self.trans.dist(trans_logits)
            #     inputs_step_dyn['obs'] = trans_dist.sample()
            #     loss_tran = self.loss_likelihood(trans_dist, rep_target)
            #     # loss_tran = self.loss_diff(inputs_step_dyn['obs'], rep_target)
            # gradients = tape_trans.gradient(loss_tran, self.trans.trainable_variables) # self.rep.trainable_variables +
            # self.trans.optimizer['trans'].apply_gradients(zip(gradients, self.trans.trainable_variables)) # self.rep.trainable_variables +
            # loss_trans = loss_trans.write(step, tf.expand_dims(tf.math.reduce_mean(loss_tran, axis=0), axis=0))
            # dyn_img_obs = inputs_step_dyn['obs']


            # ## _dyn1
            # inputs_step_dyn = {'obs':inputs_step['obs'], 'actions':action, 'step_size':1}
            # with tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done:
            # # with tape_reward, tape_done:
            #     # trans_logits = self.trans(inputs_step_dyn); trans_dist = self.trans.dist(trans_logits)
            #     trans_logits = self.trans(inputs_step_dyn, store_memory=False); trans_dist = self.trans.dist(trans_logits)
            #     inputs_step_dyn['obs'] = trans_dist.sample()
            # # dyn_img_obs = inputs_step_dyn['obs']
            # # obs_trans = obs_trans.write(step, inputs_step_dyn['obs'])

            # # obs_trans = obs_trans.write(step, self.latent_zero)
            # # obs_trans_updt = obs_trans.stack()
            # # obs_trans_updt = obs_trans_updt + inputs_step_dyn['obs']
            # # obs_trans = obs_trans.unstack(obs_trans_updt)

            # # with tf.GradientTape() as tape_reward:
            # with tape_reward:
            #     rwd_logits = self.rwd(inputs_step_dyn); rwd_dist = self.rwd.dist[0](rwd_logits[0])
            #     loss_reward = self.loss_likelihood(rwd_dist, inputs['rewards'])
            # gradients = tape_reward.gradient(loss_reward, self.rwd.trainable_variables) # self.rep.trainable_variables + self.trans.trainable_variables +
            # self.rwd.optimizer['rwd'].apply_gradients(zip(gradients, self.rwd.trainable_variables)) # self.rep.trainable_variables + self.trans.trainable_variables +
            # loss_rewards = loss_rewards.write(step, loss_reward)

            # # with tf.GradientTape() as tape_done:
            # with tape_done:
            #     done_logits = self.done(inputs_step_dyn); done_dist = self.done.dist[0](done_logits[0])
            #     loss_done = self.loss_likelihood(done_dist, inputs['dones'])
            # gradients = tape_done.gradient(loss_done, self.done.trainable_variables) # self.rep.trainable_variables + self.trans.trainable_variables +
            # self.done.optimizer['done'].apply_gradients(zip(gradients, self.done.trainable_variables)) # self.rep.trainable_variables + self.trans.trainable_variables +
            # loss_dones = loss_dones.write(step, loss_done)


            # ## _dyn7
            # rep_logits = self.rep(inputs, step=step+1, store_memory=False); rep_dist = self.rep.dist(rep_logits)
            # rep_target = rep_dist.sample()
            # inputs_step_dyn = {'obs':inputs_step['obs'], 'actions':action, 'step_size':1}
            # with tf.GradientTape() as tape_trans:
            # # with tape_trans:
            #     trans_logits = self.trans(inputs_step_dyn); trans_dist = self.trans.dist(trans_logits)
            #     dyn_img_obs = trans_dist.sample()
            #     loss_tran = self.loss_likelihood(trans_dist, rep_target)
            #     # loss_tran = self.loss_diff(dyn_img_obs, rep_target)
            # gradients = tape_trans.gradient(loss_tran, self.trans.trainable_variables) # self.rep.trainable_variables +
            # self.trans.optimizer['trans'].apply_gradients(zip(gradients, self.trans.trainable_variables)) # self.rep.trainable_variables +
            # loss_trans = loss_trans.write(step, tf.expand_dims(tf.math.reduce_mean(loss_tran, axis=0), axis=0))


            # if inputs['dones'][-1][0]:
            #     outputs = {}
            #     out_obs, out_actions, out_returns = [None]*self.obs_spec_len, [None]*self.action_spec_len, [None]*self.attn_img_scales
            #     for i in range(self.obs_spec_len): out_obs[i] = obs[i].stack()
            #     for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
            #     for i in range(self.attn_img_scales): out_returns[i] = returns[i].stack()
            #     outputs['obs'], outputs['actions'], outputs['returns'], outputs['rewards'], outputs['dones'] = out_obs, out_actions, out_returns, rewards.stack(), dones.stack()

            #     self.reset_states(); loss_act = self.MU4_dyn_act_learner(outputs, gen)

            # if step > 4 and (step%4 == 0 or inputs['dones'][-1][0]):
            #     test = rewards.stack()[-8:-4] + returns[-1].stack()[-8:-4]
            #     print()

            # if step >= 4:


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

            inputs_step = {'obs':obs, 'actions':action, 'step_size':1}
            with tf.GradientTape(persistent=True) as tape_act, tf.GradientTape(persistent=True) as tape_PG, tf.GradientTape(persistent=True) as tape_trans, tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done:
                rep_logits = self.rep(inputs_step, step=step); rep_dist = self.rep.dist(rep_logits)
                inputs_step['obs'] = rep_dist.sample()

            inputs_act = {'obs':inputs_step['obs'], 'actions':return_step, 'step_size':1}
            with tape_act:
                actin_logits = self.actin(inputs_act); actin_dist = self.actin.dist(actin_logits)
                inputs_act['obs'] = actin_dist.sample()
                action_logits = self.actout(inputs_act)
                action_dist = [None]*self.action_spec_len
                for i in range(self.action_spec_len): action_dist[i] = self.actout.dist[i](action_logits[i])
                loss_action = self.loss_likelihood(action_dist, action)
            gradients = tape_act.gradient(loss_action, self.rep.trainable_variables) # + self.actin.trainable_variables + self.actout.trainable_variables
            self.rep.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables)) # + self.actin.trainable_variables + self.actout.trainable_variables
            loss_act = loss_act.write(step, loss_action)

            if gen == 0:
                with tape_PG:
                    action_logits = self.action(inputs_step)
                    action_dist = [None]*self.action_spec_len
                    for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
                    loss_action = self.loss_PG(action_dist, action, return_step)
                gradients = tape_PG.gradient(loss_action, self.rep.trainable_variables) # + self.action.trainable_variables
                self.rep.optimizer['PG'].apply_gradients(zip(gradients, self.rep.trainable_variables)) # + self.action.trainable_variables
                loss_PG = loss_PG.write(step, loss_action)
            if gen == 2:
                with tape_PG:
                    action_logits = self.actionL(inputs_step)
                    action_dist = [None]*self.action_spec_len
                    for i in range(self.action_spec_len): action_dist[i] = self.actionL.dist[i](action_logits[i])
                    loss_action = self.loss_PG(action_dist, action, return_step)
                gradients = tape_PG.gradient(loss_action, self.rep.trainable_variables) # + self.actionL.trainable_variables
                self.rep.optimizer['PGL'].apply_gradients(zip(gradients, self.rep.trainable_variables)) # + self.actionL.trainable_variables
                loss_PG = loss_PG.write(step, loss_action)


            # obs = [None]*self.obs_spec_len
            # for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step+1:step+2]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
            # inputs_step_next = {'obs':obs}
            # rep_logits = self.rep(inputs_step_next, step=step+1, store_memory=False); rep_dist = self.rep.dist(rep_logits)
            # rep_target = rep_dist.sample()

            # with tape_trans, tape_reward, tape_done:
            #     trans_logits = self.trans(inputs_step); trans_dist = self.trans.dist(trans_logits)
            #     inputs_step['obs'] = trans_dist.sample()
            #     loss_tran = self.loss_likelihood(trans_dist, rep_target)
            #     # loss_tran = self.loss_diff(inputs_step['obs'], rep_target)
            # gradients = tape_trans.gradient(loss_tran, self.rep.trainable_variables)
            # self.rep.optimizer['trans'].apply_gradients(zip(gradients, self.rep.trainable_variables)) # + self.trans.trainable_variables
            # loss_trans = loss_trans.write(step, tf.expand_dims(tf.math.reduce_mean(loss_tran, axis=0), axis=0))

            # with tape_reward:
            #     rwd_logits = self.rwd(inputs_step); rwd_dist = self.rwd.dist[0](rwd_logits[0])
            #     loss_reward = self.loss_likelihood(rwd_dist, inputs['rewards'][step:step+1])
            # gradients = tape_reward.gradient(loss_reward, self.rep.trainable_variables) # + self.trans.trainable_variables + self.rwd.trainable_variables
            # self.rep.optimizer['rwd'].apply_gradients(zip(gradients, self.rep.trainable_variables)) # + self.trans.trainable_variables + self.rwd.trainable_variables
            # loss_rewards = loss_rewards.write(step, loss_reward)

            # with tape_done:
            #     done_logits = self.done(inputs_step); done_dist = self.done.dist[0](done_logits[0])
            #     loss_done = self.loss_likelihood(done_dist, inputs['dones'][step:step+1])
            # gradients = tape_done.gradient(loss_done, self.rep.trainable_variables) # + self.trans.trainable_variables + self.done.trainable_variables
            # self.rep.optimizer['done'].apply_gradients(zip(gradients, self.rep.trainable_variables)) # + self.trans.trainable_variables + self.done.trainable_variables
            # loss_dones = loss_dones.write(step, loss_done)


        loss['act'], loss['PG'] = loss_act.concat(), loss_PG.concat()
        loss['trans'], loss['reward'], loss['done'] = loss_trans.concat(), loss_rewards.concat(), loss_dones.concat()
        return loss

    def MU4_act_PG_learner(self, inputs, gen, return_goal, training=True):
        print("tracing -> GeneralAI MU4_PG_act_learner")
        loss = {}
        loss_act = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_PG = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        episode_len = tf.shape(inputs['dones'])[0]
        for step in tf.range(episode_len):
            obs = [None]*self.obs_spec_len
            for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
            return_step = inputs['returns'][-1][step:step+1]

            inputs_step = {'obs':obs, 'step_size':1}
            rep_logits = self.rep(inputs_step, step=step); rep_dist = self.rep.dist(rep_logits)
            inputs_step['obs'] = rep_dist.sample()

            # TODO learn this in MU4_actor using predicted reward goal
            inputs_act = {'obs':inputs_step['obs'], 'actions':return_step, 'step_size':1}
            with tf.GradientTape() as tape_act:
                actin_logits = self.actin(inputs_act); actin_dist = self.actin.dist(actin_logits)
                inputs_act['obs'] = actin_dist.sample()
                action_logits = self.actout(inputs_act)
                action_dist = [None]*self.action_spec_len
                for i in range(self.action_spec_len): action_dist[i] = self.actout.dist[i](action_logits[i])
                loss_action = self.loss_likelihood(action_dist, action)
                # loss_action = self.loss_PG(action_dist, action, return_step, returns_target=return_goal) # _lRt
                # loss_action = self.loss_PG(action_dist, action, return_step) # _lR
            gradients = tape_act.gradient(loss_action, self.actin.trainable_variables + self.actout.trainable_variables)
            self.actout.optimizer['act'].apply_gradients(zip(gradients, self.actin.trainable_variables + self.actout.trainable_variables))
            loss_act = loss_act.write(step, loss_action)
            # return_goal -= inputs['rewards'][step:step+1]; return_goal.set_shape((1,1))

            if gen == 0:
                with tf.GradientTape() as tape_PG:
                    action_logits = self.action(inputs_step)
                    action_dist = [None]*self.action_spec_len
                    for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
                    loss_action = self.loss_PG(action_dist, action, return_step)
                gradients = tape_PG.gradient(loss_action, self.action.trainable_variables)
                self.action.optimizer['action'].apply_gradients(zip(gradients, self.action.trainable_variables))
                loss_PG = loss_PG.write(step, loss_action)
            if gen == 2:
                with tf.GradientTape() as tape_PG:
                    action_logits = self.actionL(inputs_step)
                    action_dist = [None]*self.action_spec_len
                    for i in range(self.action_spec_len): action_dist[i] = self.actionL.dist[i](action_logits[i])
                    loss_action = self.loss_PG(action_dist, action, return_step)
                gradients = tape_PG.gradient(loss_action, self.actionL.trainable_variables)
                self.actionL.optimizer['action'].apply_gradients(zip(gradients, self.actionL.trainable_variables))
                loss_PG = loss_PG.write(step, loss_action)

        loss['act'], loss['PG'] = loss_act.concat(), loss_PG.concat()
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

    #         inputs_step = {'obs':obs, 'step_size':1}
    #         rep_logits = self.rep(inputs_step, step=step); rep_dist = self.rep.dist(rep_logits)
    #         inputs_step['obs'] = rep_dist.sample()

    #         inputs_act = {'obs':inputs_step['obs'], 'actions':return_step, 'step_size':1}
    #         with tf.GradientTape() as tape_act:
    #             actin_logits = self.actin(inputs_act); actin_dist = self.actin.dist(actin_logits)
    #             inputs_act['obs'] = actin_dist.sample()
    #             action_logits = self.actout(inputs_act)
    #             action_dist = [None]*self.action_spec_len
    #             for i in range(self.action_spec_len): action_dist[i] = self.actout.dist[i](action_logits[i])
    #             loss_action = self.loss_likelihood(action_dist, action)
    #             # loss_action = self.loss_PG(action_dist, action, return_step, returns_target=return_goal) # _lRt
    #             # loss_action = self.loss_PG(action_dist, action, return_step) # _lR
    #         gradients = tape_act.gradient(loss_action, self.actin.trainable_variables + self.actout.trainable_variables)
    #         self.actout.optimizer['act'].apply_gradients(zip(gradients, self.actin.trainable_variables + self.actout.trainable_variables))
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

    #         inputs_step = {'obs':obs, 'step_size':1}
    #         rep_logits = self.rep(inputs_step, step=step); rep_dist = self.rep.dist(rep_logits)
    #         inputs_step['obs'] = rep_dist.sample()

    #         with tf.GradientTape() as tape_PG:
    #             action_logits = self.action(inputs_step)
    #             action_dist = [None]*self.action_spec_len
    #             for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
    #             loss_action = self.loss_PG(action_dist, action, return_step)
    #         gradients = tape_PG.gradient(loss_action, self.action.trainable_variables)
    #         self.action.optimizer['action'].apply_gradients(zip(gradients, self.action.trainable_variables))
    #         loss_PG = loss_PG.write(step, loss_action)

    #     loss['PG'] = loss_PG.concat()
    #     return loss

    def MU4_dyn_learner3(self, inputs, training=True):
        print("tracing -> GeneralAI MU4_dyn_learner3")
        loss = {}
        loss_trans_ret = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_rewards_ret = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_dones_ret = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        episode_len = tf.shape(inputs['dones'])[0]

        obs_rep = tf.TensorArray(self.latent_spec['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])
        obs_rep_ret = [None]*self.attn_img_scales
        for i in range(self.attn_img_scales): obs_rep_ret[i] = tf.TensorArray(self.latent_spec['dtype'], size=0, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])

        for step in tf.range(episode_len+1):
            obs = [None]*self.obs_spec_len
            for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])

            inputs_step = {'obs':obs}
            rep_logits = self.rep(inputs_step, step=step); rep_dist = self.rep.dist(rep_logits)
            inputs_step['obs'] = rep_dist.sample()

            obs_rep = obs_rep.write(step, inputs_step['obs'])

            for step_scale in range(self.attn_img_scales):
                obs_rep_ret[step_scale] = obs_rep_ret[step_scale].write(step, self.latent_zero)
                returns_updt = obs_rep_ret[step_scale].stack()
                if step_scale != self.attn_img_scales-1:
                    step_size = self.attn_img_step_sizes[step_scale]
                    returns_temp = returns_updt[-step_size:] + inputs_step['obs']
                    returns_updt = tf.concat([returns_updt[:-step_size], returns_temp], axis=0)
                else: returns_updt = returns_updt + inputs_step['obs']
                obs_rep_ret[step_scale] = obs_rep_ret[step_scale].unstack(returns_updt)

        out_obs_rep_ret = [None]*self.attn_img_scales
        for i in range(self.attn_img_scales): out_obs_rep_ret[i] = obs_rep_ret[i].stack()
        inputs['obs_rep'], inputs['obs_rep_ret'] = obs_rep.stack(), out_obs_rep_ret

        for step in tf.range(episode_len):
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])

            step_size_max = episode_len - step
            inputs_step_ret = {'obs':inputs['obs_rep'][step:step+1][0], 'actions':action, 'step_size':1}
            for step_scale in range(self.attn_img_scales):
                step_size, target_done = self.attn_img_step_sizes[step_scale], tf.constant([[False]])
                if step_size >= step_size_max: step_size, target_done = step_size_max, tf.constant([[True]])
                if step_scale == self.attn_img_scales-1: target_done = tf.constant([[True]])
                target_trans, target_reward = inputs['obs_rep_ret'][step_scale][step+1:step+2][0], inputs['returns'][step_scale][step:step+1]

                inputs_step_ret['step_size'] = step_size
                loss_tran, loss_reward, loss_done = self.train_trans(inputs_step_ret, target_trans, target_reward, target_done, store_memory=False)
                loss_trans_ret = loss_trans_ret.write(step, tf.expand_dims(tf.math.reduce_mean(loss_tran, axis=0), axis=0))
                loss_rewards_ret = loss_rewards_ret.write(step, loss_reward)
                loss_dones_ret = loss_dones_ret.write(step, loss_done)

                inputs_step_ret['obs'] = inputs['obs_rep_ret'][step_scale][step:step+1][0]
                loss_tran, loss_reward, loss_done = self.train_trans(inputs_step_ret, target_trans, target_reward, target_done, store_memory=False)
                loss_trans_ret = loss_trans_ret.write(step, tf.expand_dims(tf.math.reduce_mean(loss_tran, axis=0), axis=0))
                loss_rewards_ret = loss_rewards_ret.write(step, loss_reward)
                loss_dones_ret = loss_dones_ret.write(step, loss_done)

            inputs_step_store = {'obs':inputs['obs_rep'][step:step+1][0], 'actions':action, 'step_size':1}
            trans_logits = self.trans(inputs_step_store); trans_dist = self.trans.dist(trans_logits)
            inputs_step_store['obs'] = trans_dist.sample()
            rwd_logits = self.rwd(inputs_step_store); done_logits = self.done(inputs_step_store)

        loss['trans_ret'], loss['reward_ret'], loss['done_ret'] = loss_trans_ret.concat(), loss_rewards_ret.concat(), loss_dones_ret.concat()
        return loss


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
    #         inputs_step = {'obs':inputs['obs_rep'][step:step+1][0], 'actions':self.action_zero_out, 'step_size':1}


    #         loss_PG_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_act_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_trans_ret_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_rewards_ret_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_dones_ret_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_trans_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_rewards_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_dones_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #         self.action.reset_states(use_img=True); self.actin.reset_states(use_img=True); self.actout.reset_states(use_img=True)
    #         self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
    #         inputs_step_img = {'obs':inputs_step['obs'], 'actions':self.action_zero_out, 'step_size':1}
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
    #             #         loss_action = self.loss_PG(action_dist, action, return_step)
    #             #     gradients = tape_PG.gradient(loss_action, self.action.trainable_variables)
    #             #     self.action.optimizer['action'].apply_gradients(zip(gradients, self.action.trainable_variables))
    #             #     loss_PG_img_accu = loss_PG_img_accu.write(step_img, loss_action)

    #             # inputs_act = {'obs':inputs_step_img['obs'], 'actions':return_step, 'step_size':1}
    #             # with tf.GradientTape() as tape_act_img:
    #             #     actin_logits = self.actin(inputs_act, use_img=True); actin_dist = self.actin.dist(actin_logits)
    #             #     inputs_act['obs'] = actin_dist.sample()
    #             #     action_logits = self.actout(inputs_act, use_img=True)
    #             #     action_dist = [None]*self.action_spec_len
    #             #     for i in range(self.action_spec_len): action_dist[i] = self.actout.dist[i](action_logits[i])
    #             #     loss_action = self.loss_likelihood(action_dist, action)
    #             # gradients = tape_act_img.gradient(loss_action, self.actin.trainable_variables + self.actout.trainable_variables)
    #             # self.actout.optimizer['act'].apply_gradients(zip(gradients, self.actin.trainable_variables + self.actout.trainable_variables))
    #             # loss_act_img_accu = loss_act_img_accu.write(step_img, loss_action)


    #             step_scale, step_size_max, step_size = -1, episode_len - step_img, self.attn_img_step_sizes[-1]
    #             step_size = step_size_max if step_size >= step_size_max else step_size
    #             inputs_step_img_ret = {'obs':inputs_step_img['obs'], 'actions':action, 'step_size':step_size}

    #             with tf.GradientTape() as tape_trans, tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done:
    #                 trans_logits = self.trans(inputs_step_img_ret, store_memory=False, use_img=True); trans_dist = self.trans.dist(trans_logits)
    #                 inputs_step_img_ret['obs'] = trans_dist.sample()
    #                 loss_tran = self.loss_likelihood(trans_dist, inputs['obs_rep_ret'][step_scale][step_img+1:step_img+2][0])
    #                 # loss_tran = self.loss_diff(inputs_step_img_ret['obs'], inputs['obs_rep_ret'][step_scale][step_img+1:step_img+2][0])
    #             gradients = tape_trans.gradient(loss_tran, self.trans.trainable_variables)
    #             self.trans.optimizer['trans'].apply_gradients(zip(gradients, self.trans.trainable_variables))
    #             loss_trans_ret_accu = loss_trans_ret_accu.write(step, tf.expand_dims(tf.math.reduce_mean(loss_tran, axis=0), axis=0))

    #             with tape_reward:
    #                 rwd_logits = self.rwd(inputs_step_img_ret, store_memory=False, use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #                 loss_reward = self.loss_likelihood(rwd_dist, inputs['returns'][step_scale][step_img:step_img+1])
    #             gradients = tape_reward.gradient(loss_reward, self.trans.trainable_variables + self.rwd.trainable_variables) # + self.rwd.trainable_variables
    #             self.rwd.optimizer['rwd'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.rwd.trainable_variables)) # + self.rwd.trainable_variables
    #             loss_rewards_ret_accu = loss_rewards_ret_accu.write(step_img, loss_reward)

    #             with tape_done:
    #                 done_logits = self.done(inputs_step_img_ret, store_memory=False, use_img=True); done_dist = self.done.dist[0](done_logits[0])
    #                 loss_done = self.loss_likelihood(done_dist, tf.constant([[True]]))
    #             gradients = tape_done.gradient(loss_done, self.trans.trainable_variables + self.done.trainable_variables) # + self.done.trainable_variables
    #             self.done.optimizer['done'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.done.trainable_variables)) # + self.done.trainable_variables
    #             loss_dones_ret_accu = loss_dones_ret_accu.write(step_img, loss_done)


    #             inputs_step_img['actions'] = action
    #             with tf.GradientTape() as tape_trans, tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done:
    #                 trans_logits = self.trans(inputs_step_img, use_img=True); trans_dist = self.trans.dist(trans_logits)
    #                 inputs_step_img['obs'] = trans_dist.sample()
    #                 loss_tran = self.loss_likelihood(trans_dist, inputs['obs_rep'][step_img+1:step_img+2][0])
    #                 # loss_tran = self.loss_diff(inputs_step_img['obs'], inputs['obs_rep'][step_img+1:step_img+2][0])
    #             gradients = tape_trans.gradient(loss_tran, self.trans.trainable_variables)
    #             self.trans.optimizer['trans'].apply_gradients(zip(gradients, self.trans.trainable_variables))
    #             loss_trans_accu = loss_trans_accu.write(step, tf.expand_dims(tf.math.reduce_mean(loss_tran, axis=0), axis=0))

    #             with tape_reward:
    #                 rwd_logits = self.rwd(inputs_step_img, use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #                 loss_reward = self.loss_likelihood(rwd_dist, inputs['rewards'][step_img:step_img+1])
    #             gradients = tape_reward.gradient(loss_reward, self.trans.trainable_variables + self.rwd.trainable_variables) # + self.rwd.trainable_variables
    #             self.rwd.optimizer['rwd'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.rwd.trainable_variables)) # + self.rwd.trainable_variables
    #             loss_rewards_accu = loss_rewards_accu.write(step_img, loss_reward)

    #             with tape_done:
    #                 done_logits = self.done(inputs_step_img, use_img=True); done_dist = self.done.dist[0](done_logits[0])
    #                 loss_done = self.loss_likelihood(done_dist, inputs['dones'][step_img:step_img+1])
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

    #         inputs_step_store = {'obs':inputs_step['obs'], 'actions':action, 'step_size':1}
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
    #         #         loss_action = self.loss_PG(action_dist, action, return_step)
    #         #     gradients = tape_PG.gradient(loss_action, self.action.trainable_variables)
    #         #     self.action.optimizer['action'].apply_gradients(zip(gradients, self.action.trainable_variables))
    #         #     loss_PG = loss_PG.write(step, loss_action)

    #         # inputs_act = {'obs':inputs_step['obs'], 'actions':return_step, 'step_size':1}
    #         # with tf.GradientTape() as tape_act:
    #         #     actin_logits = self.actin(inputs_act, use_img=True, store_real=True); actin_dist = self.actin.dist(actin_logits)
    #         #     inputs_act['obs'] = actin_dist.sample()
    #         #     action_logits = self.actout(inputs_act, use_img=True, store_real=True)
    #         #     action_dist = [None]*self.action_spec_len
    #         #     for i in range(self.action_spec_len): action_dist[i] = self.actout.dist[i](action_logits[i])
    #         #     loss_action = self.loss_likelihood(action_dist, action)
    #         # gradients = tape_act.gradient(loss_action, self.actin.trainable_variables + self.actout.trainable_variables) # self.rep.trainable_variables +
    #         # self.actout.optimizer['act'].apply_gradients(zip(gradients, self.actin.trainable_variables + self.actout.trainable_variables)) # self.rep.trainable_variables +
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

    #         inputs_step = {'obs':obs, 'actions':self.action_zero_out, 'step_size':1}
    #         with tf.GradientTape(persistent=True) as tape_act:
    #             rep_logits = self.rep(inputs_step, step=step); rep_dist = self.rep.dist(rep_logits)
    #             inputs_step['obs'] = rep_dist.sample()



    #         # loss_PG_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         # loss_act_img_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_rewards_ret_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_dones_ret_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_rewards_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
    #         loss_dones_accu = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

    #         self.action.reset_states(use_img=True); self.actin.reset_states(use_img=True); self.actout.reset_states(use_img=True)
    #         self.trans.reset_states(use_img=True); self.rwd.reset_states(use_img=True); self.done.reset_states(use_img=True)
    #         inputs_step_img = {'obs':inputs_step['obs'], 'actions':self.action_zero_out, 'step_size':1}
    #         for step_img in tf.range(step, episode_len):
    #             action = [None]*self.action_spec_len
    #             for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step_img:step_img+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
    #             return_step = inputs['returns'][-1][step_img:step_img+1]

    #             # if gen == 0:
    #             #     with tf.GradientTape() as tape_PG:
    #             #         action_logits = self.action(inputs_step_img, use_img=True)
    #             #         action_dist = [None]*self.action_spec_len
    #             #         for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
    #             #         loss_action = self.loss_PG(action_dist, action, return_step)
    #             #     gradients = tape_PG.gradient(loss_action, self.action.trainable_variables)
    #             #     self.action.optimizer['action'].apply_gradients(zip(gradients, self.action.trainable_variables))
    #             #     # loss_PG_img_accu = loss_PG_img_accu.write(step_img, loss_action)

    #             # inputs_act = {'obs':inputs_step_img['obs'], 'actions':return_step, 'step_size':1}
    #             # with tf.GradientTape() as tape_act_img:
    #             #     actin_logits = self.actin(inputs_act, use_img=True); actin_dist = self.actin.dist(actin_logits)
    #             #     inputs_act['obs'] = actin_dist.sample()
    #             #     action_logits = self.actout(inputs_act, use_img=True)
    #             #     action_dist = [None]*self.action_spec_len
    #             #     for i in range(self.action_spec_len): action_dist[i] = self.actout.dist[i](action_logits[i])
    #             #     loss_action = self.loss_likelihood(action_dist, action)
    #             # gradients = tape_act_img.gradient(loss_action, self.actin.trainable_variables + self.actout.trainable_variables)
    #             # self.actout.optimizer['act'].apply_gradients(zip(gradients, self.actin.trainable_variables + self.actout.trainable_variables))
    #             # # loss_act_img_accu = loss_act_img_accu.write(step_img, loss_action)


    #             step_scale = self.attn_img_scales-1
    #             step_size = tf.math.pow(self.attn_mem_multi, step_scale)
    #             inputs_step_img_ret = {'obs':inputs_step_img['obs'], 'actions':action, 'step_size':step_size}
    #             with tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done:
    #                 trans_logits = self.trans(inputs_step_img_ret, store_memory=False, use_img=True); trans_dist = self.trans.dist(trans_logits)
    #                 inputs_step_img_ret['obs'] = trans_dist.sample()

    #             with tape_reward:
    #                 rwd_logits = self.rwd(inputs_step_img_ret, store_memory=False, use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #                 loss_reward = self.loss_likelihood(rwd_dist, inputs['returns'][step_scale][step_img:step_img+1])
    #             gradients = tape_reward.gradient(loss_reward, self.trans.trainable_variables) # + self.rwd.trainable_variables
    #             self.rwd.optimizer['rwd'].apply_gradients(zip(gradients, self.trans.trainable_variables)) # + self.rwd.trainable_variables
    #             loss_rewards_ret_accu = loss_rewards_ret_accu.write(step_img, loss_reward)

    #             with tape_done:
    #                 done_logits = self.done(inputs_step_img_ret, store_memory=False, use_img=True); done_dist = self.done.dist[0](done_logits[0])
    #                 loss_done = self.loss_likelihood(done_dist, tf.constant([[True]]))
    #             gradients = tape_done.gradient(loss_done, self.trans.trainable_variables) # + self.done.trainable_variables
    #             self.done.optimizer['done'].apply_gradients(zip(gradients, self.trans.trainable_variables)) # + self.done.trainable_variables
    #             loss_dones_ret_accu = loss_dones_ret_accu.write(step_img, loss_done)


    #             inputs_step_img['actions'] = action
    #             with tf.GradientTape(persistent=True) as tape_reward, tf.GradientTape(persistent=True) as tape_done:
    #                 trans_logits = self.trans(inputs_step_img, use_img=True); trans_dist = self.trans.dist(trans_logits)
    #                 inputs_step_img['obs'] = trans_dist.sample()

    #             with tape_reward:
    #                 rwd_logits = self.rwd(inputs_step_img, use_img=True); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #                 loss_reward = self.loss_likelihood(rwd_dist, inputs['rewards'][step_img:step_img+1])
    #             gradients = tape_reward.gradient(loss_reward, self.trans.trainable_variables) # + self.rwd.trainable_variables
    #             self.rwd.optimizer['rwd'].apply_gradients(zip(gradients, self.trans.trainable_variables)) # + self.rwd.trainable_variables
    #             loss_rewards_accu = loss_rewards_accu.write(step_img, loss_reward)

    #             with tape_done:
    #                 done_logits = self.done(inputs_step_img, use_img=True); done_dist = self.done.dist[0](done_logits[0])
    #                 loss_done = self.loss_likelihood(done_dist, inputs['dones'][step_img:step_img+1])
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
    #         step_size = tf.math.pow(self.attn_mem_multi, step_scale)
    #         inputs_step_store = {'obs':inputs_step['obs'], 'actions':action, 'step_size':step_size}
    #         trans_logits = self.trans(inputs_step_store, store_memory=False); trans_dist = self.trans.dist(trans_logits)
    #         inputs_step_store['obs'] = trans_dist.sample()
    #         rwd_logits = self.rwd(inputs_step_store, store_memory=False); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #         loss_return = self.loss_likelihood(rwd_dist, inputs['returns'][step_scale][step:step+1])

    #         inputs_step_store = {'obs':inputs_step['obs'], 'actions':action, 'step_size':1}
    #         trans_logits = self.trans(inputs_step_store); trans_dist = self.trans.dist(trans_logits)
    #         inputs_step_store['obs'] = trans_dist.sample()
    #         rwd_logits = self.rwd(inputs_step_store); rwd_dist = self.rwd.dist[0](rwd_logits[0])
    #         loss_reward = self.loss_likelihood(rwd_dist, inputs['rewards'][step:step+1])
    #         done_logits = self.done(inputs_step_store); done_dist = self.done.dist[0](done_logits[0])
    #         loss_done = self.loss_likelihood(done_dist, inputs['dones'][step:step+1])


    #         if gen == 0:
    #             with tf.GradientTape() as tape_PG:
    #                 action_logits = self.action(inputs_step, use_img=True, store_real=True)
    #                 action_dist = [None]*self.action_spec_len
    #                 for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
    #                 loss_action = self.loss_PG(action_dist, action, return_step)
    #                 # surprise = -loss_return-loss_reward-loss_done
    #                 # surprise = loss_return+loss_reward+loss_done
    #                 # loss_action = self.loss_PG(action_dist, action, surprise)
    #             gradients = tape_PG.gradient(loss_action, self.action.trainable_variables)
    #             self.action.optimizer['action'].apply_gradients(zip(gradients, self.action.trainable_variables))
    #             loss_PG = loss_PG.write(step, loss_action)

    #         # inputs_act = {'obs':inputs_step['obs'], 'actions':return_step, 'step_size':1}
    #         # with tape_act:
    #         #     actin_logits = self.actin(inputs_act, use_img=True, store_real=True); actin_dist = self.actin.dist(actin_logits)
    #         #     inputs_act['obs'] = actin_dist.sample()
    #         #     action_logits = self.actout(inputs_act, use_img=True, store_real=True)
    #         #     action_dist = [None]*self.action_spec_len
    #         #     for i in range(self.action_spec_len): action_dist[i] = self.actout.dist[i](action_logits[i])
    #         #     loss_action = self.loss_likelihood(action_dist, action)
    #         #     # surprise = -loss_return-loss_reward-loss_done
    #         #     # surprise = loss_return+loss_reward+loss_done
    #         #     # loss_action = self.loss_PG(action_dist, action, surprise)
    #         # gradients = tape_act.gradient(loss_action, self.rep.trainable_variables + self.actin.trainable_variables + self.actout.trainable_variables)
    #         # self.actout.optimizer['act'].apply_gradients(zip(gradients, self.rep.trainable_variables + self.actin.trainable_variables + self.actout.trainable_variables))
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
    #                 loss_reward = self.loss_likelihood(rwd_dist, inputs['rewards'][step_img:step_img+1])
    #             gradients = tape_reward.gradient(loss_reward, self.trans.trainable_variables + self.rwd.trainable_variables)
    #             self.rwd.optimizer['rwd'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.rwd.trainable_variables))
    #             loss_rewards_img_accu = loss_rewards_img_accu.write(step_img, loss_reward)

    #             with tape_done:
    #                 done_logits = self.done(inputs_step, use_img=True); done_dist = self.done.dist[0](done_logits[0])
    #                 loss_done = self.loss_likelihood(done_dist, inputs['dones'][step_img:step_img+1])
    #             gradients = tape_done.gradient(loss_done, self.trans.trainable_variables + self.done.trainable_variables)
    #             self.done.optimizer['done'].apply_gradients(zip(gradients, self.trans.trainable_variables + self.done.trainable_variables))
    #             loss_dones_img_accu = loss_dones_img_accu.write(step_img, loss_done)

    #         loss_rewards = loss_rewards.write(step, tf.math.reduce_mean(loss_rewards_img_accu.stack(), axis=0))
    #         loss_dones = loss_dones.write(step, tf.math.reduce_mean(loss_dones_img_accu.stack(), axis=0))

    #     loss['reward'], loss['done'] = loss_rewards.concat(), loss_dones.concat()
    #     return loss

    def MU4(self):
        print("tracing -> GeneralAI MU4")
        num_gen = 3
        ma, std = tf.constant(0,tf.float64), tf.constant(0,self.compute_dtype)
        episode, stop = tf.constant(0), tf.constant(False)
        while episode < self.max_episodes*num_gen and not stop:
            tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}

            gen, episode_gen = episode%num_gen, episode//num_gen
            log_metrics, train = [True,True,True,True,True,True,True,True,True,True,True,True,True,True], True
            return_goal = tf.constant([[200.0]], tf.float64) # TODO try changing return_goal each episode to incrementally increase above current total returns
            # return_goal_alt = tf.constant([[10.0]], tf.float64)
            return_goal_alt = tf.random.uniform((1,1), minval=0.0, maxval=200.0, dtype=tf.float64)
            if gen == 0: return_goal, log_metrics, train, gen = return_goal, [False,False,False,False,False,False,False,True,True,True,True,True,False,False], True, 0 # action/PG
            if gen == 1: return_goal, log_metrics, train, gen = return_goal, [True,True,True,True,False,False,False,False,False,False,False,False,False,False], False, 1 # actout/act
            if gen == 2: return_goal, log_metrics, train, gen = return_goal, [False,False,False,False,True,True,True,False,False,False,False,False,False,False], True, 2 # random, actionL/PGL
            if gen == 3: return_goal, log_metrics, train, gen = return_goal_alt, [False,False,False,False,False,False,False,False,False,False,False,False,False,False], True, 1 # act alt


            self.reset_states(); outputs, inputs, loss_actor = self.MU4_actor(inputs, gen, return_goal, return_goal_alt)
            rewards_total = outputs['returns'][-1][0][0]
            if gen == 0: util.stats_update(self.action.stats_rwd, rewards_total, tf.float64); ma, _, _, _ = util.stats_get(self.action.stats_rwd, self.float64_eps, tf.float64)
            if gen == 1: util.stats_update(self.actout.stats_rwd, rewards_total, tf.float64); ma, _, _, _ = util.stats_get(self.actout.stats_rwd, self.float64_eps, tf.float64)
            if gen == 2: util.stats_update(self.actionL.stats_rwd, rewards_total, tf.float64); ma, _, _, _ = util.stats_get(self.actionL.stats_rwd, self.float64_eps, tf.float64)


            loss_rep = {'PG':tf.constant([0], self.compute_dtype), 'act':tf.constant([0], self.compute_dtype)}
            loss_act = {'PG':tf.constant([0], self.compute_dtype), 'act':tf.constant([0], self.compute_dtype)}
            loss_PG = {'PG':tf.constant([0], self.compute_dtype)}
            loss_dyn = {'trans':tf.constant([0], self.compute_dtype), 'reward':tf.constant([0], self.compute_dtype), 'done':tf.constant([0], self.compute_dtype)}
            if train:
                self.reset_states(); loss_rep = self.MU4_rep_learner(outputs, gen) # _repL1
                # self.reset_states(); loss_act = self.MU4_act_learner(outputs, return_goal)
                self.reset_states(); loss_act = self.MU4_act_PG_learner(outputs, gen, return_goal)
                # if gen == 0: self.reset_states(); loss_PG = self.MU4_PG_learner(outputs)
                # self.reset_states(); loss_dyn = self.MU3_dyn_learner(outputs) # _dyn3
                # self.reset_states(); loss_dyn = self.MU4_dyn_learner(outputs) # _dyn4
                # self.reset_states(); loss_dyn = self.MU4_dyn_act_learner(outputs, gen) # _dyn5
                # self.reset_states(); loss_dyn = self.MU4_dyn_learner2(outputs, gen) # _dyn8
                # self.reset_states(); loss_dyn = self.MU4_dyn_learner3(outputs) # _dyn9

                if gen == 0:
                    util.stats_update(self.action.stats_loss, tf.math.reduce_mean(loss_act['PG']), self.compute_dtype); ma_loss, _, _, std = util.stats_get(self.action.stats_loss, self.float_eps, self.compute_dtype)
                    maL, _, _, _ = util.stats_get(self.actionL.stats_rwd, self.float64_eps, tf.float64)
                    if self.action.stats_loss['iter'] > 10 and std < 1.0 and tf.math.abs(ma_loss) < 1.0:
                        if ma > maL: util.net_copy(self.action, self.actionL)
                        util.net_reset(self.action)
                        # self.action.optimizer['action'].learning_rate = tf.random.uniform((), dtype=tf.float64, maxval=2e-4, minval=self.float64_eps)
                if gen == 1: util.stats_update(self.actout.stats_loss, tf.math.reduce_mean(loss_act['act']), self.compute_dtype); _, _, _, std = util.stats_get(self.actout.stats_loss, self.float_eps, self.compute_dtype)
                if gen == 2: util.stats_update(self.actionL.stats_loss, tf.math.reduce_mean(loss_act['PG']), self.compute_dtype); _, _, _, std = util.stats_get(self.actionL.stats_loss, self.float_eps, self.compute_dtype)


            metrics = [log_metrics, episode_gen, ma, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                ma, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0],
                ma, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0],
                # tf.math.reduce_mean(loss_actor['returns_pred']),
                tf.math.reduce_mean(loss_act['PG']), std,
                # tf.math.reduce_mean(loss_act['PG']), std,
                # tf.math.reduce_mean(loss_act['act']),
                # tf.math.reduce_mean(loss_rep['trans']),
                # tf.math.reduce_mean(loss_rep['reward']), tf.math.reduce_mean(loss_rep['done']),
                # tf.math.reduce_mean(loss_actor['trans']), tf.math.reduce_mean(loss_actor['trans_img']),
                # tf.math.reduce_mean(loss_actor['reward']), tf.math.reduce_mean(loss_actor['done']),
                # tf.math.reduce_mean(loss_actor['reward_img']), tf.math.reduce_mean(loss_actor['done_img']),
                # tf.math.reduce_mean(loss_dyn['trans_ret']), tf.math.reduce_mean(loss_dyn['trans']),
                # tf.math.reduce_mean(loss_dyn['reward_ret']), tf.math.reduce_mean(loss_dyn['done_ret']),
                # tf.math.reduce_mean(loss_dyn['reward']), tf.math.reduce_mean(loss_dyn['done']),
                # tf.math.reduce_mean(loss_act['reward']), tf.math.reduce_mean(loss_act['done']),
                # tf.math.reduce_mean(loss_actor['entropy']),
            ]
            if self.trader: metrics += [tf.math.reduce_mean(tf.concat([outputs['obs'][3],inputs['obs'][3]],0)), inputs['obs'][3][-1][0],
                tf.math.reduce_mean(tf.concat([outputs['obs'][4],inputs['obs'][4]],0)), tf.math.reduce_mean(tf.concat([outputs['obs'][5],inputs['obs'][5]],0)),
                inputs['obs'][0][-1][0] - outputs['obs'][0][0][0],]
            dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

            stop = tf.numpy_function(self.check_stop, [tf.constant(0)], tf.bool); stop.set_shape(())
            episode += 1




def params(): pass
load_model, save_model = False, False
max_episodes = 10
learn_rate = 2e-6 # 5 = testing, 6 = more stable/slower
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
arch = 'MU4' # Dreamer/planner w/imagination+generalization
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

        name = "{}-{}-a{}-{}{}".format(name, machine, device, time.strftime("%y-%m-%d-%H-%M-%S"), extra)
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
