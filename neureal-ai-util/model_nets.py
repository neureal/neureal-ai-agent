from collections import OrderedDict
import numpy as np
import tensorflow as tf
import model_util as util


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
