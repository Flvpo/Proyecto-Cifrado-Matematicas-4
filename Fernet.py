import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
import pyaudio
import wave
import hashlib
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
import os
import json
from datetime import datetime
from pathlib import Path

class VoiceEncryption:
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024
        self.RECORD_SECONDS = 3
        self.N_SAMPLES = 5
        
        # Crear estructura de carpetas
        self.base_dir = Path("voice_encryption_system")
        self.encrypted_dir = self.base_dir / "encrypted_files"
        self.decrypted_dir = self.base_dir / "decrypted_files"
        self.keys_dir = self.base_dir / "keys"
        self.voice_samples_dir = self.base_dir / "voice_samples"
        self.logs_dir = self.base_dir / "logs"
        
        self.create_directory_structure()
        
    def create_directory_structure(self):
        """Crea la estructura de carpetas necesaria."""
        directories = [
            self.base_dir,
            self.encrypted_dir,
            self.decrypted_dir,
            self.keys_dir,
            self.voice_samples_dir,
            self.logs_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Crear archivo README
        readme_path = self.base_dir / "README.md"
        if not readme_path.exists():
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write("""# Sistema de Cifrado Basado en Voz

Estructura de carpetas:
- encrypted_files: Archivos cifrados
- decrypted_files: Archivos descifrados
- keys: Claves y parámetros de cifrado
- voice_samples: Muestras de voz temporales
- logs: Registros de operaciones

Para usar el sistema, ejecute main.py""")
    
    def log_operation(self, operation_type, filename):
        """Registra las operaciones en el archivo de log."""
        log_file = self.logs_dir / "operations.log"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {operation_type}: {filename}\n"
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
    
    def record_audio(self, sample_number):
        """Graba una muestra de audio y la guarda temporalmente."""
        audio = pyaudio.PyAudio()
        
        print(f"Grabando muestra {sample_number + 1}/5...")
        
        stream = audio.open(format=self.FORMAT,
                          channels=self.CHANNELS,
                          rate=self.RATE,
                          input=True,
                          frames_per_buffer=self.CHUNK)
        
        frames = []
        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(np.frombuffer(data, dtype=np.float32))
            
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        signal = np.concatenate(frames)
        
        # Guardar muestra temporal
        sample_path = self.voice_samples_dir / f"sample_{sample_number}.npy"
        np.save(sample_path, signal)
        
        return signal
    
    def preprocess_signal(self, signal):
        """Preprocesa la señal de audio."""
        signal = signal / np.max(np.abs(signal))
        signal = signal - np.mean(signal)
        return signal
    
    def extract_features(self, signal):
        """Extrae características espectrales usando FFT."""
        window = np.hamming(len(signal))
        signal_windowed = signal * window
        fft_result = fft(signal_windowed)
        magnitude_spectrum = np.abs(fft_result[:len(fft_result)//2])
        magnitude_spectrum = magnitude_spectrum / np.max(magnitude_spectrum)
        
        n_features = 100
        peak_indices = np.argsort(magnitude_spectrum)[-n_features:]
        features = magnitude_spectrum[peak_indices]
        
        return features
    
    def generate_key_from_features(self, features):
        """Genera una clave criptográfica a partir de las características espectrales."""
        features_bytes = features.tobytes()
        salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = kdf.derive(features_bytes)
        return key, salt
    
    def create_voice_key(self):
        """Crea una clave basada en múltiples muestras de voz."""
        all_features = []
        
        for i in range(self.N_SAMPLES):
            signal = self.record_audio(i)
            processed_signal = self.preprocess_signal(signal)
            features = self.extract_features(processed_signal)
            all_features.append(features)
            
        average_features = np.mean(all_features, axis=0)
        key, salt = self.generate_key_from_features(average_features)
        fernet_key = Fernet.generate_key()
        
        # Limpiar muestras temporales
        for sample_file in self.voice_samples_dir.glob("sample_*.npy"):
            sample_file.unlink()
            
        return key, salt, fernet_key
    
    def encrypt_file(self, input_file):
        """Cifra un archivo usando la clave derivada de voz."""
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {input_file}")
            
        # Generar nombres de archivo
        encrypted_filename = input_path.stem + "_encrypted" + input_path.suffix
        encrypted_path = self.encrypted_dir / encrypted_filename
        key_filename = input_path.stem + "_key.json"
        key_path = self.keys_dir / key_filename
        
        # Crear clave basada en voz
        key, salt, fernet_key = self.create_voice_key()
        
        # Guardar información de la clave
        key_info = {
            "salt": salt.hex(),
            "fernet_key": fernet_key.decode(),
            "original_filename": input_path.name,
            "encryption_date": datetime.now().isoformat()
        }
        
        with open(key_path, "w") as f:
            json.dump(key_info, f, indent=4)
        
        # Leer y cifrar archivo
        with open(input_path, "rb") as f:
            data = f.read()
        
        fernet = Fernet(fernet_key)
        encrypted_data = fernet.encrypt(data)
        
        # Guardar archivo cifrado
        with open(encrypted_path, "wb") as f:
            f.write(encrypted_data)
            
        self.log_operation("Cifrado", input_path.name)
        
        return encrypted_path
    
    def decrypt_file(self, encrypted_file):
        """Descifra un archivo usando una nueva muestra de voz."""
        encrypted_path = Path(encrypted_file)
        if not encrypted_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo cifrado: {encrypted_file}")
            
        # Encontrar archivo de clave correspondiente
        key_filename = encrypted_path.stem.replace("_encrypted", "") + "_key.json"
        key_path = self.keys_dir / key_filename
        
        if not key_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de clave: {key_filename}")
        
        # Leer información de la clave
        with open(key_path, "r") as f:
            key_info = json.load(f)
            
        salt = bytes.fromhex(key_info["salt"])
        fernet_key = key_info["fernet_key"].encode()
        
        # Obtener nueva muestra de voz
        all_features = []
        for i in range(self.N_SAMPLES):
            signal = self.record_audio(i)
            processed_signal = self.preprocess_signal(signal)
            features = self.extract_features(processed_signal)
            all_features.append(features)
        
        average_features = np.mean(all_features, axis=0)
        
        # Generar clave con la misma sal
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = kdf.derive(average_features.tobytes())
        
        # Generar nombre para archivo descifrado
        decrypted_filename = encrypted_path.stem.replace("_encrypted", "_decrypted") + encrypted_path.suffix
        decrypted_path = self.decrypted_dir / decrypted_filename
        
        # Leer y descifrar archivo
        with open(encrypted_path, "rb") as f:
            encrypted_data = f.read()
        
        fernet = Fernet(fernet_key)
        decrypted_data = fernet.decrypt(encrypted_data)
        
        # Guardar archivo descifrado
        with open(decrypted_path, "wb") as f:
            f.write(decrypted_data)
            
        self.log_operation("Descifrado", encrypted_path.name)
        
        # Limpiar muestras temporales
        for sample_file in self.voice_samples_dir.glob("sample_*.npy"):
            sample_file.unlink()
            
        return decrypted_path

def main():
    """Función principal de ejemplo."""
    encryptor = VoiceEncryption()
    
    while True:
        print("\nSistema de Cifrado por Voz")
        print("1. Cifrar archivo")
        print("2. Descifrar archivo")
        print("3. Salir")
        
        option = input("\nSeleccione una opción: ")
        
        if option == "1":
            input_file = input("Ingrese la ruta del archivo a cifrar: ")
            try:
                encrypted_file = encryptor.encrypt_file(input_file)
                print(f"\nArchivo cifrado guardado en: {encrypted_file}")
            except Exception as e:
                print(f"Error al cifrar: {e}")
                
        elif option == "2":
            encrypted_file = input("Ingrese la ruta del archivo cifrado: ")
            try:
                decrypted_file = encryptor.decrypt_file(encrypted_file)
                print(f"\nArchivo descifrado guardado en: {decrypted_file}")
            except Exception as e:
                print(f"Error al descifrar: {e}")
                
        elif option == "3":
            break
            
        else:
            print("Opción no válida")

if __name__ == "__main__":
    main()
