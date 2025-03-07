import numpy as np
import wave
import struct

# Define parameters for the beep sound
sample_rate = 44100  # CD quality
duration = 0.3  # 300ms
frequency = 880  # 880 Hz (A5)
volume = 0.7  # Maximum volume (adjust if needed)

# Generate a sine wave
t = np.linspace(0, duration, int(sample_rate * duration), False)
wave_data = np.sin(2 * np.pi * frequency * t) * volume

# Add fade in/out to avoid clicking
fade_duration = 0.05  # 50ms fade in/out
fade_samples = int(fade_duration * sample_rate)
fade_in = np.linspace(0, 1, fade_samples)
fade_out = np.linspace(1, 0, fade_samples)

wave_data[:fade_samples] *= fade_in
wave_data[-fade_samples:] *= fade_out

# Convert to 16-bit values
wave_data = (wave_data * 32767).astype(np.int16)

# Save as WAV file
with wave.open('alarm_beep.wav', 'w') as wav_file:
    # Set parameters
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 2 bytes = 16 bits
    wav_file.setframerate(sample_rate)
    
    # Write data
    for sample in wave_data:
        wav_file.writeframes(struct.pack('h', sample))

print("Alarm sound created successfully: alarm_beep.wav")