
from conf import *
import numpy as np
import cv2
from scipy.io import wavfile
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


def straight_paper_img_to_single_line(img, margin, pixel_height):
    img = np.array(img)
    w, h = img.shape[1] - margin * 2, img.shape[0] - margin * 2

    track_number = h // pixel_height // 2
    audio_line = np.zeros((track_number*w, pixel_height*2))

    for track_i in range(track_number):
        vertical_part_from = margin + track_i * pixel_height * 2
        vertical_part_to = vertical_part_from + pixel_height * 2

        img_part = img[vertical_part_from:vertical_part_to, margin:margin+w, 0]
        img_part = np.swapaxes(img_part, 0, 1)

        audio_line[track_i*w:(track_i+1)*w] = img_part

    return audio_line


def straight_paper_img_to_audio_signal(audio_line, upscale, pixel_height, max_amplitude):
    audio_signal = np.zeros(audio_line.shape[0] * upscale)

    for i in tqdm(range(audio_line.shape[0])):
        choice = np.array([])
        probs = np.array([])

        for j in range(audio_line.shape[1]):
            if audio_line[i][j] < 255:
                back_interpolated_wave = int((j - pixel_height) / pixel_height * max_amplitude)
                choice = np.append(choice, back_interpolated_wave)
                probs = np.append(probs, 255 - audio_line[i][j])

        if choice.shape[0] > 0:
            probs /= np.sum(probs)
            audio_signal[i*upscale:i*upscale+upscale] = np.random.choice(choice, p=probs, size=upscale)
            #audio_signal += [np.mean(choice)] * UPSCALE

    return audio_signal


if __name__ == "__main__":
    print('Processing picture to audio')

    img = np.array(cv2.imread('res.png'))

    audio_line_encrypt = straight_paper_img_to_single_line(img, MARGIN, HEIGHT)
    audio_signal = straight_paper_img_to_audio_signal(audio_line_encrypt, UPSCALE, HEIGHT, HIGH_F)
    print(audio_signal.shape, audio_signal)

    audio_signal_pcm = np.int16(audio_signal)

    wavfile.write('output.wav', FREQUENCY * UPSCALE, audio_signal_pcm)

    print('Finish!!!')
