
from conf import *
import wave
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import math


def signal_to_data_array(signal_arr, orig_freq, new_freq, max_amplitude, pixel_height):
    time_length = len(signal_arr) / orig_freq
    frames_n_with_new_freq = math.ceil(time_length * new_freq)
    spec_arr = np.zeros((frames_n_with_new_freq, pixel_height*2), dtype=np.float64)

    #for i in tqdm(range(signal_arr.shape[0])):
    for i in tqdm(range(1000000)):
        w = signal_arr[i]
        if w >= max_amplitude or w <= -max_amplitude: continue

        timeline_index = int(i / orig_freq * new_freq)
        wave_interpolated = int(w / max_amplitude * pixel_height)

        spec_arr[timeline_index][wave_interpolated + pixel_height] += 1

    # interpolated to contrast with the preservation of proportions
    spec_arr *= 255 / np.mean(spec_arr) / orig_freq * new_freq * 100

    return spec_arr


def apply_sound_data_to_img(img, sound_data, pixel_height, margin):
    img_arr = np.array(img)
    w, h = img_arr.shape[1] - margin*2, img_arr.shape[0] - margin*2

    sound_data = np.array_split(sound_data, math.ceil(len(sound_data) / w))

    if len(sound_data) * pixel_height * 2 > h:
        print('Waring: not enough space')
        print(f'it\'s possible only to {round(h / len(sound_data) / pixel_height / 2 * 100, 1)}%')
        sound_data = sound_data[:h // 2 // pixel_height]

    for i in tqdm(range(len(sound_data))):
        img_like_sound = np.expand_dims(sound_data[i], axis=-1)
        img_like_sound = np.repeat(img_like_sound, 3, axis=-1)
        img_like_sound = np.swapaxes(img_like_sound, 0, 1)

        from_y = i * pixel_height * 2 + margin
        to_y = from_y + pixel_height * 2

        img_arr[from_y:to_y, margin:margin+img_like_sound.shape[1], :] -= img_like_sound.astype('uint8')

    return img_arr


if __name__ == "__main__":
    print('Processing audio to picture')

    wav_obj = wave.open('sounds/fly me to the moon.wav', 'rb')

    sample_freq = wav_obj.getframerate()
    samples_n = wav_obj.getnframes()
    channels_n = wav_obj.getnchannels()
    audio_t = samples_n / sample_freq

    print('Frequency:', sample_freq)
    print('Duration:', audio_t)
    print('Number of channels:', channels_n)

    signal_wave = wav_obj.readframes(samples_n)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    l_channel = signal_array[0::2]
    r_channel = signal_array[1::2]

    #plt.specgram(l_channel, Fs=sample_freq, vmin=-10, vmax=50)
    #plt.colorbar()
    #plt.show()

    data_to_print = signal_to_data_array(l_channel, sample_freq, FREQUENCY, HIGH_F, HEIGHT)
    print('Contrast:', np.mean(data_to_print))

    img_layout = np.array(cv2.imread('layout.png'))
    res_img = apply_sound_data_to_img(img_layout, data_to_print, HEIGHT, MARGIN)
    cv2.imwrite('res.png', res_img)

    print('Finish!!!')


