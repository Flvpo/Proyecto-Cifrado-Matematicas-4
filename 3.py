import numpy as np
import pyaudio
import wave
import scipy.fft
from pycryptodome.Cipher import AES
from pycryptodome.Random import get_random_bytes
from scipy.io import wavfile
import hashlib
import statistics

class VoiceEncryption:
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024
        self.RECORD_SECONDS = 3
        self.NUM_SAMPLES = 5
        
    def record_audio_samples(self):
        """Graba múltiples muestras de voz y retorna sus espectros de frecuencia"""
        print(f"Se grabarán {self.NUM_SAMPLES} muestras de voz...")
        frequency_features = []
        
        p = pyaudio.PyAudio()
        
        for i in range(self.NUM_SAMPLES):
            print(f"\nGrabando muestra {i+1}/{self.NUM_SAMPLES}")
            print("Hable ahora...")
            
            stream = p.open(format=self.FORMAT,
                          channels=self.CHANNELS,
                          rate=self.RATE,
                          input=True,
                          frames_per_buffer=self.CHUNK)
            
            frames = []
            for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                data = stream.read(self.CHUNK)
                frames.append(np.frombuffer(data, dtype=np.float32))
            
            stream.stop_stream()
            stream.close()
            
            # Concatenar frames y aplicar FFT
            audio_signal = np.concatenate(frames)
            spectrum = self.extract_frequency_features(audio_signal)
            frequency_features.append(spectrum)
            
            print("Muestra grabada correctamente")
        
        p.terminate()
        return frequency_features
    
    def extract_frequency_features(self, audio_signal):
        """Extrae características frecuenciales usando FFT"""
        # Aplicar ventana Hamming para reducir efectos de borde
        windowed_signal = audio_signal * np.hamming(len(audio_signal))
        
        # Calcular FFT
        fft_spectrum = scipy.fft.fft(windowed_signal)
        frequencies = scipy.fft.fftfreq(len(fft_spectrum), 1/self.RATE)
        
        # Usar solo la mitad positiva del espectro
        positive_frequencies_mask = frequencies > 0
        positive_frequencies = frequencies[positive_frequencies_mask]
        positive_spectrum = np.abs(fft_spectrum[positive_frequencies_mask])
        
        # Normalizar espectro
        normalized_spectrum = positive_spectrum / np.max(positive_spectrum)
        
        # Extraer características relevantes (picos principales)
        peaks = self.find_peaks(normalized_spectrum, threshold=0.3)
        return peaks
    
    def find_peaks(self, spectrum, threshold):
        """Encuentra picos significativos en el espectro"""
        peaks = []
        for i in range(1, len(spectrum)-1):
            if (spectrum[i] > spectrum[i-1] and 
                spectrum[i] > spectrum[i+1] and 
                spectrum[i] > threshold):
                peaks.append(spectrum[i])
        return sorted(peaks)[:10]  # Usar los 10 picos más significativos
    
    def generate_key_from_samples(self, frequency_features):
        """Genera una clave a partir de múltiples muestras de características frecuenciales"""
        # Calcular la mediana de cada característica a través de las muestras
        feature_medians = []
        for i in range(min(len(features) for features in frequency_features)):
            values_at_i = [features[i] for features in frequency_features]
            feature_medians.append(statistics.median(values_at_i))
        
        # Convertir características en bytes para generar la clave
        features_bytes = np.array(feature_medians).tobytes()
        key = hashlib.sha256(features_bytes).digest()
        return key
    
    def encrypt_file(self, input_file, output_file):
        """Cifra un archivo usando la voz como clave"""
        # Grabar y procesar muestras de voz
        frequency_features = self.record_audio_samples()
        key = self.generate_key_from_samples(frequency_features)
        
        # Generar IV aleatorio
        iv = get_random_bytes(16)
        
        # Crear cifrador AES
        cipher = AES.new(key, AES.MODE_CBC, iv)
        
        # Leer y cifrar archivo
        with open(input_file, 'rb') as f:
            data = f.read()
            # Padding PKCS7
            padding_length = 16 - (len(data) % 16)
            data += bytes([padding_length]) * padding_length
            encrypted_data = cipher.encrypt(data)
        
        # Guardar IV y datos cifrados
        with open(output_file, 'wb') as f:
            f.write(iv)
            f.write(encrypted_data)
    
    def decrypt_file(self, input_file, output_file):
        """Descifra un archivo usando la voz como clave"""
        # Grabar y procesar muestras de voz
        frequency_features = self.record_audio_samples()
        key = self.generate_key_from_samples(frequency_features)
        
        # Leer IV y datos cifrados
        with open(input_file, 'rb') as f:
            iv = f.read(16)
            encrypted_data = f.read()
        
        # Crear descifrador AES
        cipher = AES.new(key, AES.MODE_CBC, iv)
        
        # Descifrar datos y quitar padding
        decrypted_data = cipher.decrypt(encrypted_data)
        padding_length = decrypted_data[-1]
        decrypted_data = decrypted_data[:-padding_length]
        
        # Guardar archivo descifrado
        with open(output_file, 'wb') as f:
            f.write(decrypted_data)

# Ejemplo de uso
encryptor = VoiceEncryption()