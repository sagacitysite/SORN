import numpy as np
import pyaudio
import math

winsound.Beep(17000, 100)

transitions = []
iterate = np.arange(0.1, 0.51, 0.025)
for it in iterate:
    transitions.append([[1-(2*it), it, 0, it],
                        [0.5, 0, 0.5, 0],
                        [0, 0.5, 0, 0.5],
                        [0.5, 0, 0.5, 0]])
transitions = np.array(transitions)

size = np.shape(transitions)[1]

first_transition = transitions[0]
last_transition = transitions[16]

def create_sequence(transition):
    sequence = [0] # start in state 0
    sequence_length = 40
    for l in range(sequence_length):
        sequence.append(np.random.choice(size, 1, p=transition[sequence[-1]].flatten())[0])

    return sequence

def play_sequence(sequence):
    for seq in sequence:
        playTone((seq+1)*200, 0.2)

play_sequence(create_sequence(first_transition))
play_sequence(create_sequence(last_transition))

PyAudio = pyaudio.PyAudio

def playTone( freq , length):

    bit_rate = 16000 #number of frames per second/frameset.

    frequency = freq #in Hz, waves per second
    play_time = length #in seconds to play sound

    if frequency > bit_rate:
        bit_rate = frequency+100

    num_frames = int(bit_rate * play_time)
    total_frames = num_frames % bit_rate
    wave_info = ''

    for x in xrange(num_frames):
     wave_info = wave_info+chr(int(math.sin(x/((bit_rate/frequency)/math.pi))*127+128))

    for x in xrange(total_frames):
     wave_info = wave_info+chr(128)

    p = PyAudio()
    stream = p.open(format = p.get_format_from_width(1),
                    channels = 1,
                    rate = bit_rate,
                    output = True)

    stream.write(wave_info)
    stream.stop_stream()
    stream.close()
    p.terminate()