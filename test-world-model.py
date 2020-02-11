import numpy as np
np.set_printoptions(precision=8, suppress=True)
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading, time, os
import talib


data_size = 4

data_streams = []
for i in range(data_size): # random sines
    freq = np.random.uniform(0.001, 0.4)
    ampl = np.random.uniform(0.1, 1.0)
    data_streams.append((freq, ampl))
def get_data(i=0, len=1):
    l = []
    for j in range(data_size):
        freq, ampl = data_streams[j]
        l.append(np.sin(np.arange(i,i+len) * freq) * ampl)
    npl = np.stack(l,axis=-1)
    return ((npl + 1.0) / 2.0 * 256.0).astype(np.uint8) # uint8 (range 0:255)

def _pack_data(data): # convert 0:255 --> -1.0:+1.0
    return ((data - 127.5) / 128.0).astype(np.float32) # quantized
def _unpack_data(data): # convert -1.0:+1.0 --> 0:255
    data = np.around(data * 128.0 + 127.5)
    np.place(data, data>255.0, 255.0)
    return data.astype(np.uint8) # threshold




batch_size = 1
history_timesteps = 1
future_timesteps = 1
model_input_shape = (batch_size, history_timesteps, data_size) # batchsize, timesteps, features
model_output_shape = (batch_size, future_timesteps, data_size) # batchsize, timesteps, features
output_size = batch_size * future_timesteps * data_size


## models
# representer
# generator
# discriminator
# actors
# 

generator_model = tf.keras.models.Sequential()
generator_model.add(tf.keras.layers.LSTM(256,
    batch_input_shape=model_input_shape,
    return_sequences=True,
    stateful=True
))
generator_model.add(tf.keras.layers.Dense(output_size))
generator_model.compile(optimizer='sgd', loss='mae')
generator_model.summary()

discriminator_model = tf.keras.models.Sequential()
discriminator_model.add(tf.keras.layers.LSTM(256,
    batch_input_shape=model_input_shape,
    return_sequences=True,
    stateful=True
))
discriminator_model.add(tf.keras.layers.Dense(output_size))
discriminator_model.compile(optimizer='sgd', loss='mae')
discriminator_model.summary()

exit()

ema_buff = 200
num_timesteps = 10000

plt_window_size = num_timesteps
plt_window_interval = int(np.ceil(num_timesteps/10000)) # (18 inch wide window = about 2000 pixels, so 10000 = 5 samples per pixel)
plt_lines = {'loss':{'color':'purple', 'alpha':0.2}, 'loss EMA':{'color':'purple', 'alpha':1.0}} # orange, green

epochs = 1

plt_window = np.arange(-plt_window_size + 1, 1, plt_window_interval)
for _,line in plt_lines.items(): line['history'] = np.zeros(int(np.ceil(plt_window_size/plt_window_interval)), dtype=np.float32)
plt_fig, plt_axs = plt.subplots(1, 1, figsize=(18, 8), tight_layout=True, num='test') # must be first, creates window
plt_axs.set(title='World Model', ylabel='value', xlabel='timestep')
for name,line in plt_lines.items(): line['line'], = plt_axs.plot(plt_window, line['history'], color=line['color'], alpha=line['alpha'], label=name)
plt_axs.legend(loc='upper left')
plt_axs.set_ylim(0.0, 0.6)
plt_axs.grid(True)

def plt_animate(i):
    for _,line in plt_lines.items(): line['line'].set_ydata(line['history'])
plt_ani = animation.FuncAnimation(plt_fig, plt_animate, init_func=None, interval=500, blit=False)

quit = False
def loop():

    data_old = get_data(i=0)
    data_old_packed = _pack_data(data_old)

    loss_sample = np.zeros(ema_buff, dtype=np.float64)
    for i in range(1, num_timesteps):
        if quit: break
        data = get_data(i=i)
        data_packed = _pack_data(data)

        fit_history = generator_model.fit(data_old_packed.reshape(model_input_shape), data_packed.reshape(model_output_shape), epochs=epochs, validation_split=0.0, verbose=0)

        eval = fit_history.history['loss'][-1]
        # print("test {} type {}".format(test, type(test)))
        loss_sample = np.roll(loss_sample, -1, 0)
        loss_sample[-1] = eval
        # print("test type {} shape {} dtype {}\n{}".format(type(test), test.shape, test.dtype, test))

        if i % plt_window_interval == 0.0:
            for _,line in plt_lines.items(): line['history'] = np.roll(line['history'], -1, 0)
            plt_lines['loss']['history'][-1] = eval
            plt_lines['loss EMA']['history'][-1] = talib.EMA(loss_sample, timeperiod=ema_buff)[-1]

        data_old_packed = data_packed
        # time.sleep(1.1)


t = threading.Thread(target=loop)
t.start()
plt.show()
quit = True
t.join()
