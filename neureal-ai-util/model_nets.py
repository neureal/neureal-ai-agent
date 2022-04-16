from collections import OrderedDict
import numpy as np
import tensorflow as tf
import model_util as util


class ArchFull(tf.keras.Model):
    def __init__(self, name, inputs, opt_spec, stats_spec, spec_in, spec_out, latent_spec, net_blocks=1, net_attn=False, net_lstm=False, net_attn_io=False, num_heads=1, memory_size=None, aug_data_pos=False):
        super(ArchFull, self).__init__(name=name)
        net_attn_io_out = False
        self.inp = In(latent_spec, spec_in, net_attn_io=net_attn_io, num_heads=num_heads, aug_data_pos=aug_data_pos)
        self.net = Net(latent_spec, net_blocks=net_blocks, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=num_heads, memory_size=memory_size)
        self.out = Out(latent_spec, spec_out, net_attn_io=net_attn_io_out)
        self.dist = self.out.dist

        self.optimizer = OrderedDict()
        for spec in opt_spec: self.optimizer[spec['name']] = util.optimizer(name, spec)
        self.stats = OrderedDict()
        for spec in stats_spec: self.stats[spec['name']] = {'b1':tf.constant(spec['b1'],tf.float64), 'b1_n':tf.constant(1-spec['b1'],tf.float64), 'b2':tf.constant(spec['b2'],tf.float64), 'b2_n':tf.constant(1-spec['b2'],tf.float64),
            'ma':tf.Variable(0, dtype=tf.float64, trainable=False, name='{}/stats_{}/ma'.format(name,spec['name'])), 'ema':tf.Variable(0, dtype=tf.float64, trainable=False, name='{}/stats_{}/ema'.format(name,spec['name'])),
            'iter':tf.Variable(0, dtype=tf.float64, trainable=False, name='{}/stats_{}/iter'.format(name,spec['name'])),}

        self(inputs); self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        arch_in = "{}D{}~{}".format(('Aio+' if net_attn_io else ''), latent_spec['inp'], self.inp.arch_in)
        arch_net = "{:02d}{}{}D{}".format(net_blocks, ('AT+' if net_attn else ''), ('LS+' if net_lstm else ''), latent_spec['midp'])
        arch_out = "{}D{}~{}".format(('Aio+' if net_attn_io_out else ''), latent_spec['outp'], self.out.arch_out)
        self.arch_desc = "{}[in{}-net{}-out{}-{}]".format(name, arch_in, arch_net, arch_out, self.inp.arch_lat)

    def reset_states(self, use_img=False):
        for layer in self.net.layer_attn: layer.reset_states(use_img=use_img)
        for layer in self.net.layer_lstm: layer.reset_states()
    def call(self, inputs, store_memory=True, use_img=False, store_real=False, training=None):
        out = self.inp(inputs, training=training)
        out = self.net(out, store_memory=store_memory, use_img=use_img, store_real=store_real, training=training)
        dist = self.net.dist(out); out = dist.sample()
        out = self.out(out, training=training)

        for out_n in out:
            isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out_n), tf.math.is_inf(out_n)))
            if isinfnan > 0: tf.print(self.name, 'net out:', out)
        return out

class ArchTrans(tf.keras.Model):
    def __init__(self, name, inputs, opt_spec, stats_spec, spec_in, latent_spec, net_blocks=1, net_attn=False, net_lstm=False, net_attn_io=False, num_heads=1, memory_size=None, aug_data_pos=False):
        super(ArchTrans, self).__init__(name=name)
        self.inp = In(latent_spec, spec_in, obs_latent=True, net_attn_io=net_attn_io, num_heads=num_heads, aug_data_pos=aug_data_pos)
        self.net = Net(latent_spec, net_blocks=net_blocks, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=num_heads, memory_size=memory_size)
        self.dist = self.net.dist

        self.optimizer = OrderedDict()
        for spec in opt_spec: self.optimizer[spec['name']] = util.optimizer(name, spec)
        self.stats = OrderedDict()
        for spec in stats_spec: self.stats[spec['name']] = {'b1':tf.constant(spec['b1'],tf.float64), 'b1_n':tf.constant(1-spec['b1'],tf.float64), 'b2':tf.constant(spec['b2'],tf.float64), 'b2_n':tf.constant(1-spec['b2'],tf.float64),
            'ma':tf.Variable(0, dtype=tf.float64, trainable=False, name='{}/stats_{}/ma'.format(name,spec['name'])), 'ema':tf.Variable(0, dtype=tf.float64, trainable=False, name='{}/stats_{}/ema'.format(name,spec['name'])),
            'iter':tf.Variable(0, dtype=tf.float64, trainable=False, name='{}/stats_{}/iter'.format(name,spec['name'])),}

        self(inputs); self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        arch_in = "{}D{}~{}".format(('Aio+' if net_attn_io else ''), latent_spec['inp'], self.inp.arch_in)
        arch_net = "{:02d}{}{}D{}".format(net_blocks, ('AT+' if net_attn else ''), ('LS+' if net_lstm else ''), latent_spec['midp'])
        self.arch_desc = "{}[in{}-net{}-{}]".format(name, arch_in, arch_net, self.net.arch_lat)

    def reset_states(self, use_img=False):
        for layer in self.net.layer_attn: layer.reset_states(use_img=use_img)
        for layer in self.net.layer_lstm: layer.reset_states()
    def call(self, inputs, store_memory=True, use_img=False, store_real=False, training=None):
        out = self.inp(inputs, training=training)
        out = self.net(out, store_memory=store_memory, use_img=use_img, store_real=store_real, training=training)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print(self.name, 'net out:', out)
        return out

class ArchRep(tf.keras.Model):
    def __init__(self, name, inputs, opt_spec, stats_spec, spec_in, latent_spec, net_attn_io=False, num_heads=1, aug_data_pos=False):
        super(ArchRep, self).__init__(name=name)
        self.inp = In(latent_spec, spec_in, net_attn_io=net_attn_io, num_heads=num_heads, aug_data_pos=aug_data_pos)

        self.optimizer = OrderedDict()
        for spec in opt_spec: self.optimizer[spec['name']] = util.optimizer(name, spec)
        self.stats = OrderedDict()
        for spec in stats_spec: self.stats[spec['name']] = {'b1':tf.constant(spec['b1'],tf.float64), 'b1_n':tf.constant(1-spec['b1'],tf.float64), 'b2':tf.constant(spec['b2'],tf.float64), 'b2_n':tf.constant(1-spec['b2'],tf.float64),
            'ma':tf.Variable(0, dtype=tf.float64, trainable=False, name='{}/stats_{}/ma'.format(name,spec['name'])), 'ema':tf.Variable(0, dtype=tf.float64, trainable=False, name='{}/stats_{}/ema'.format(name,spec['name'])),
            'iter':tf.Variable(0, dtype=tf.float64, trainable=False, name='{}/stats_{}/iter'.format(name,spec['name'])),}

        self(inputs); self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        arch_in = "{}D{}~{}".format(('Aio+' if net_attn_io else ''), latent_spec['inp'], self.inp.arch_in)
        self.arch_desc = "{}[in{}-{}]".format(name, arch_in, self.inp.arch_lat)

    def reset_states(self, use_img=False): return
    def call(self, inputs, training=None):
        out = self.inp(inputs, training=training)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print(self.name, 'net out:', out)
        return out

class ArchGen(tf.keras.Model):
    def __init__(self, name, inputs, opt_spec, stats_spec, spec_out, latent_spec, net_blocks=1, net_attn=False, net_lstm=False, net_attn_io=False, num_heads=1, memory_size=None):
        super(ArchGen, self).__init__(name=name)
        net_attn_io_out = False
        self.net = Net(latent_spec, net_blocks=net_blocks, net_attn=net_attn, net_lstm=net_lstm, net_attn_io=net_attn_io, num_heads=num_heads, memory_size=memory_size)
        self.out = Out(latent_spec, spec_out, net_attn_io=net_attn_io_out)
        self.dist = self.out.dist

        self.optimizer = OrderedDict()
        for spec in opt_spec: self.optimizer[spec['name']] = util.optimizer(name, spec)
        self.stats = OrderedDict()
        for spec in stats_spec: self.stats[spec['name']] = {'b1':tf.constant(spec['b1'],tf.float64), 'b1_n':tf.constant(1-spec['b1'],tf.float64), 'b2':tf.constant(spec['b2'],tf.float64), 'b2_n':tf.constant(1-spec['b2'],tf.float64),
            'ma':tf.Variable(0, dtype=tf.float64, trainable=False, name='{}/stats_{}/ma'.format(name,spec['name'])), 'ema':tf.Variable(0, dtype=tf.float64, trainable=False, name='{}/stats_{}/ema'.format(name,spec['name'])),
            'iter':tf.Variable(0, dtype=tf.float64, trainable=False, name='{}/stats_{}/iter'.format(name,spec['name'])),}

        self(inputs); self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        arch_net = "{:02d}{}{}D{}".format(net_blocks, ('AT+' if net_attn else ''), ('LS+' if net_lstm else ''), latent_spec['midp'])
        arch_out = "{}D{}~{}".format(('Aio+' if net_attn_io_out else ''), latent_spec['outp'], self.out.arch_out)
        self.arch_desc = "{}[net{}-out{}-{}]".format(name, arch_net, arch_out, self.net.arch_lat)

    def reset_states(self, use_img=False):
        for layer in self.net.layer_attn: layer.reset_states(use_img=use_img)
        for layer in self.net.layer_lstm: layer.reset_states()
    def call(self, inputs, store_memory=True, use_img=False, store_real=False, training=None):
        out = self.net(inputs, store_memory=store_memory, use_img=use_img, store_real=store_real, training=training)
        dist = self.net.dist(out); out = dist.sample()
        out = self.out(out, training=training)

        for out_n in out:
            isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out_n), tf.math.is_inf(out_n)))
            if isinfnan > 0: tf.print(self.name, 'net out:', out)
        return out



class In(tf.keras.layers.Layer):
    def __init__(self, latent_spec, spec_in, obs_latent=False, net_attn_io=False, num_heads=1, aug_data_pos=False):
        super(In, self).__init__(name='inp')
        inp, evo, latent_size, aio_max_latents = latent_spec['inp'], latent_spec['evo'], latent_spec['latent_size'], latent_spec['max_latents']
        self.obs_latent, self.num_latents, self.net_attn_io2 = obs_latent, 0, False
        self.layer_flatten = tf.keras.layers.Flatten()

        self.net_ins = len(spec_in); self.input_names, self.layer_attn_in, self.layer_mlp_in, self.pos_idx_in = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
        for i in range(self.net_ins):
            space_name, input_name, event_shape, event_size, channels, num_latents = spec_in[i]['space_name'], spec_in[i]['name'], spec_in[i]['event_shape'], spec_in[i]['event_size'], spec_in[i]['channels'], spec_in[i]['num_latents']; self.num_latents += num_latents
            if space_name not in self.input_names: self.input_names[space_name], self.layer_attn_in[space_name], self.layer_mlp_in[space_name], self.pos_idx_in[space_name] = 0, [], [], []
            self.input_names[space_name] += 1
            if aug_data_pos and event_size > 1:
                    pos_idx = np.indices(event_shape[:-1])
                    pos_idx = np.moveaxis(pos_idx, 0, -1)
                    # pos_idx = pos_idx / (np.max(pos_idx).item() / 2.0) - 1.0
                    self.pos_idx_in[space_name] += [tf.constant(pos_idx, dtype=self.compute_dtype)]
                    channels += pos_idx.shape[-1]
            else: self.pos_idx_in[space_name] += [None]
            if net_attn_io and event_size > 1:
                self.layer_attn_in[space_name] += [util.MultiHeadAttention(latent_size=latent_size, num_heads=num_heads, norm=True, hidden_size=inp, evo=evo, residual=False, cross_type=1, num_latents=num_latents, channels=channels, name='attn_in_{}_{}'.format(space_name, input_name))]
            else: self.layer_attn_in[space_name] += [None]
            self.layer_mlp_in[space_name] += [util.MLPBlock(hidden_size=inp, latent_size=latent_size, evo=evo, residual=False, name='mlp_in_{}_{}'.format(space_name, input_name))]
        if net_attn_io and self.num_latents > aio_max_latents:
            self.net_attn_io2, self.num_latents = True, aio_max_latents
            self.layer_attn_io2 = util.MultiHeadAttention(latent_size=latent_size, num_heads=num_heads, norm=False, residual=False, cross_type=1, num_latents=self.num_latents, channels=latent_size, name='attn_io2')
        if obs_latent: self.net_ins += 1; self.num_latents += latent_spec['num_latents']
        else: latent_spec.update({'num_latents':self.num_latents})

        self.arch_in, self.arch_lat = "I{}{}".format(self.net_ins, ('io2' if self.net_attn_io2 else '')), "L{}{}x{}".format(latent_spec['dist_type'], self.num_latents, latent_size)

    def call(self, inputs, training=None):
        out_accu, out_accu_i = [None]*self.net_ins, 0
        for input_name in self.input_names.keys():
            for i in range(self.input_names[input_name]):
                out = tf.cast(inputs[input_name][i], self.compute_dtype)
                # out = tf.expand_dims(out, axis=-1) # TODO try splitting down to individual scaler level
                if self.pos_idx_in[input_name][i] is not None:
                    shape = tf.concat([tf.shape(out)[0:1], self.pos_idx_in[input_name][i].shape], axis=0)
                    pos_idx = tf.broadcast_to(self.pos_idx_in[input_name][i], shape)
                    out = tf.concat([out, pos_idx], axis=-1)
                if self.layer_attn_in[input_name][i] is not None: out = self.layer_attn_in[input_name][i](out)
                else: out = self.layer_flatten(out)
                out_accu[out_accu_i] = self.layer_mlp_in[input_name][i](out); out_accu_i += 1
        if self.obs_latent: out_accu[-1] = tf.cast(inputs['obs'], self.compute_dtype)
        # out = tf.math.add_n(out_accu) # out = tf.math.accumulate_n(out_accu)
        out = tf.concat(out_accu, axis=0)
        if self.net_attn_io2: out = self.layer_attn_io2(out)
        return out

class Net(tf.keras.layers.Layer):
    def __init__(self, latent_spec, net_blocks=1, net_attn=False, net_lstm=False, net_attn_io=False, num_heads=1, memory_size=None):
        super(Net, self).__init__(name='net')
        midp, evo, latent_size = latent_spec['midp'], latent_spec['evo'], latent_spec['latent_size']
        self.net_blocks, self.net_attn, self.net_lstm, self.net_attn_io = net_blocks, net_attn, net_lstm, net_attn_io
        self.layer_flatten = tf.keras.layers.Flatten()

        self.layer_attn, self.layer_lstm, self.layer_mlp = [], [], []
        for i in range(net_blocks):
            if net_attn:
                self.layer_attn += [util.MultiHeadAttention(latent_size=latent_size, num_heads=num_heads, memory_size=memory_size, residual=True, name='attn_{:02d}'.format(i))]
                self.layer_mlp += [util.MLPBlock(hidden_size=midp, latent_size=latent_size, evo=None, residual=True, name='mlp_{:02d}'.format(i))]
            elif net_lstm:
                self.layer_lstm += [tf.keras.layers.LSTM(midp, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i))]
                self.layer_mlp += [tf.keras.layers.Dense(latent_size, name='dense_{:02d}'.format(i))]
            else: self.layer_mlp += [util.MLPBlock(hidden_size=midp, latent_size=latent_size, evo=evo, residual=False, name='mlp_{:02d}'.format(i))]

        self.num_latents = latent_spec['num_latents']
        params_size, self.dist = util.distribution(latent_spec)
        # if net_attn_io: self.layer_attn_out_logits = util.MultiHeadAttention(latent_size=params_size, num_heads=num_heads, norm=True, hidden_size=midp, evo=evo, residual=False, cross_type=1, num_latents=self.num_latents, channels=latent_size, name='attn_out_logits')
        if net_attn_io: self.layer_attn_out_logits = util.MultiHeadAttention(latent_size=params_size, num_heads=num_heads, norm=False, residual=False, cross_type=1, num_latents=self.num_latents, channels=latent_size, name='attn_out_logits')
        else: self.layer_dense_out_logits = tf.keras.layers.Dense(self.num_latents*params_size, name='dense_out_logits')

        self.arch_lat = "L{}{}x{}".format(latent_spec['dist_type'], self.num_latents, latent_size)

    def call(self, inputs, store_memory=True, use_img=False, store_real=False, training=None):
        out = tf.cast(inputs, self.compute_dtype)
        for i in range(self.net_blocks):
            if self.net_attn: out = tf.squeeze(self.layer_attn[i](tf.expand_dims(out, axis=0), auto_mask=training, store_memory=store_memory, use_img=use_img, store_real=store_real), axis=0)
            if self.net_lstm: out = tf.squeeze(self.layer_lstm[i](tf.expand_dims(out, axis=0), training=training), axis=0)
            out = self.layer_mlp[i](out)
        if self.net_attn_io: out = self.layer_attn_out_logits(out)
        else:
            out = self.layer_flatten(tf.expand_dims(out, axis=0))
            out = self.layer_dense_out_logits(out)
            out = tf.reshape(out, (self.num_latents, -1))
        return out

class Out(tf.keras.layers.Layer):
    def __init__(self, latent_spec, spec_out, net_attn_io=False, num_heads=1):
        super(Out, self).__init__(name='out')
        outp, evo, latent_size = latent_spec['outp'], latent_spec['evo'], latent_spec['latent_size']
        self.net_attn_io = net_attn_io

        self.net_outs = len(spec_out); params_size, self.dist, self.layer_attn_out, self.layer_out_logits, self.logits_step_shape, self.arch_out = [None]*self.net_outs, [None]*self.net_outs, [None]*self.net_outs, [None]*self.net_outs, [None]*self.net_outs, "O"
        for i in range(self.net_outs):
            space_name, output_name, dist_type, num_components = spec_out[i]['space_name'], spec_out[i]['name'], spec_out[i]['dist_type'], spec_out[i]['num_components']

            params_size[i], self.dist[i] = util.distribution(spec_out[i])
            # if net_attn_io: self.layer_attn_out[i] = util.MultiHeadAttention(latent_size=latent_size, num_heads=1, norm=False, residual=False, cross_type=2, num_latents=max_steps, channels=params_size[i], name='attn_out_{}_{}'.format(space_name, output_name))
            if net_attn_io: self.layer_attn_out[i] = util.MultiHeadAttention(latent_size=params_size[i], num_heads=num_heads, norm=False, residual=False, cross_type=1, num_latents=1, channels=latent_size, name='attn_out_{}_{}'.format(space_name, output_name))
            if dist_type == 'd': self.layer_out_logits[i] = tf.keras.layers.Dense(params_size[i], name='dense_out_logits_{}_{}'.format(space_name, output_name))
            else: self.layer_out_logits[i] = util.MLPBlock(hidden_size=outp, latent_size=params_size[i], evo=evo, residual=False, name='mlp_out_logits_{}_{}'.format(space_name, output_name))

            self.logits_step_shape += [tf.TensorShape([1]+[params_size[i]])]

            self.arch_out += "{}{}".format(dist_type, num_components)

    def call(self, inputs, batch_size=tf.constant(1), training=None):
        out = tf.cast(inputs, self.compute_dtype)
        if not self.net_attn_io: out = tf.reshape(out, (batch_size, -1))
        out_logits = [None]*self.net_outs
        for i in range(self.net_outs):
            out_logits[i] = out if not self.net_attn_io else self.layer_attn_out[i](out, num_latents=batch_size)
            out_logits[i] = self.layer_out_logits[i](out_logits[i])
        return out_logits
