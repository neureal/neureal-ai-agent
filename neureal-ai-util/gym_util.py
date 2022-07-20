from collections import OrderedDict
import numpy as np
import tensorflow as tf
import gym

def get_space_zero(space):
    if isinstance(space, gym.spaces.Discrete): zero = np.asarray(0, space.dtype)
    elif isinstance(space, gym.spaces.Box): zero = np.zeros(space.shape, space.dtype)
    elif isinstance(space, gym.spaces.Tuple):
        zero = [None]*len(space.spaces)
        for i,s in enumerate(space.spaces): zero[i] = get_space_zero(s)
        zero = tuple(zero)
    elif isinstance(space, gym.spaces.Dict):
        zero = OrderedDict()
        for k,s in space.spaces.items(): zero[k] = get_space_zero(s)
    return zero

# TODO add different kinds of net_type? 0 = Dense, 1 = 2 layer Dense, 2 = Conv2D, etc
def get_spec(space, space_name='obs', name='', compute_dtype='float64', net_attn_io=False, aio_max_latents=16, mixture_multi=4):
    if isinstance(space, gym.spaces.Discrete):
        dtype, dtype_out = tf.dtypes.as_dtype(space.dtype), tf.dtypes.as_dtype('int32')
        spec = [{'space_name':space_name, 'name':name, 'dtype':dtype, 'dtype_out':dtype_out, 'min':tf.constant(0,dtype_out), 'max':tf.constant(space.n-1,dtype_out),
            'dist_type':'c', 'num_components':space.n, 'event_shape':(1,), 'event_size':1, 'channels':1, 'step_shape':tf.TensorShape((1,1)), 'num_latents':1}]
        zero, zero_out = [tf.constant([[0]], dtype)], [tf.constant([[0]], dtype_out)]
    # elif isinstance(space, gym.spaces.MultiDiscrete): # TODO
    elif isinstance(space, gym.spaces.Box):
        dtype, dtype_out = tf.dtypes.as_dtype(space.dtype), tf.dtypes.as_dtype(compute_dtype)
        if dtype == tf.uint8 or dtype == tf.int32 or dtype == tf.int64 or dtype == tf.bool: dist_type, num_components, event_shape, dtype_out = 'c', int(space.high.max().item())+1, (space.shape[-1],), tf.dtypes.as_dtype('int32')
        else: dist_type, num_components, event_shape = 'mx', int(np.prod(space.shape).item()*mixture_multi), space.shape
        event_size, channels, step_shape = int(np.prod(space.shape[:-1]).item()), space.shape[-1], tf.TensorShape([1]+list(space.shape))
        num_latents = aio_max_latents if event_size > aio_max_latents else event_size
        spec = [{'space_name':space_name, 'name':name, 'dtype':dtype, 'dtype_out':dtype_out, 'min':tf.constant(space.low,dtype_out), 'max':tf.constant(space.high,dtype_out),
            'dist_type':dist_type, 'num_components':num_components, 'event_shape':event_shape, 'event_size':event_size, 'channels':channels, 'step_shape':step_shape, 'num_latents':num_latents}]
        zero, zero_out = [tf.zeros(step_shape, dtype)], [tf.zeros(step_shape, dtype_out)]
    elif isinstance(space, (gym.spaces.Tuple, gym.spaces.Dict)):
        spec, zero, zero_out = [], [], []
        loop = space.spaces.items() if isinstance(space, gym.spaces.Dict) else enumerate(space.spaces)
        for k,s in loop:
            spec_sub, zero_sub, zero_out_sub = get_spec(s, space_name=space_name, name=('' if name=='' else name+'_')+str(k), compute_dtype=compute_dtype, net_attn_io=net_attn_io, aio_max_latents=aio_max_latents, mixture_multi=mixture_multi)
            spec += spec_sub; zero += zero_sub; zero_out += zero_out_sub
    return spec, zero, zero_out


# TODO test tf.nest.flatten(data)
def space_to_feat(data, space):
    feat = []
    if isinstance(data, tuple):
        for i,v in enumerate(data): feat += space_to_feat(v, space[i])
    elif isinstance(data, dict):
        for k,v in data.items(): feat += space_to_feat(v, space[k])
    elif isinstance(data, np.ndarray): feat = [np.expand_dims(data,0)]
    else: feat = [np.asarray([[data]], space.dtype)]
    return feat

# TODO test tf.nest.pack_sequence_as(out, space)
def out_to_space(out, space, idx):
    if isinstance(space, (gym.spaces.Discrete, gym.spaces.Box)):
        data = out[idx[0]]
        if isinstance(space, gym.spaces.Box): data = np.reshape(data, space.shape)
        if isinstance(space, gym.spaces.Discrete): data = data.item() # numpy.int64 is coming in here in graph mode
        idx[0] += 1
    elif isinstance(space, gym.spaces.Tuple):
        data = [None]*len(space.spaces)
        for i,s in enumerate(space.spaces): data[i] = out_to_space(out, s, idx)
        data = tuple(data)
    elif isinstance(space, gym.spaces.Dict):
        data = OrderedDict()
        for k,s in space.spaces.items(): data[k] = out_to_space(out, s, idx)
    return data


def struc_to_feat(data):
    if data.dtype.names is None:
        if len(data.shape) == 1: data = np.expand_dims(data,-1)
        feat = [data]
    else:
        feat = []
        for k in data.dtype.names: feat += struc_to_feat(data[k])
    return feat

def out_to_struc(out, dtype):
    for i in range(len(out)): out[i] = np.frombuffer(out[i], dtype=np.uint8)
    out = np.concatenate(out)
    out = np.frombuffer(out, dtype=dtype)
    return out


def space_to_bytes(data, space):
    byts = []
    if isinstance(data, tuple):
        for i,v in enumerate(data): byts += space_to_bytes(v, space[i])
    elif isinstance(data, dict):
        for k,v in data.items(): byts += space_to_bytes(v, space[k])
    else:
        if not isinstance(data, np.ndarray): data = np.asarray(data, space.dtype)
        byts = [np.frombuffer(data, dtype=np.uint8)]
    return byts

def bytes_to_space(byts, space, idxs, idx):
    if isinstance(space, (gym.spaces.Discrete, gym.spaces.Box)):
        data = byts[idxs[idx[0]]:idxs[idx[0]+1]]
        if space.dtype != np.uint8: data = np.frombuffer(data, dtype=space.dtype)
        if isinstance(space, gym.spaces.Box): data = np.reshape(data, space.shape)
        if isinstance(space, gym.spaces.Discrete): data = data.item()
        idx[0] += 1
    elif isinstance(space, gym.spaces.Tuple):
        data = [None]*len(space.spaces)
        for i,s in enumerate(space.spaces): data[i] = bytes_to_space(byts, s, idxs, idx)
        data = tuple(data)
    elif isinstance(space, gym.spaces.Dict):
        data = OrderedDict()
        for k,s in space.spaces.items(): data[k] = bytes_to_space(byts, s, idxs, idx)
    return data
