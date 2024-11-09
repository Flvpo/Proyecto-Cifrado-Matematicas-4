from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
from scipy.fft import fft
from scipy import linalg, signal
from scipy.stats import pearsonr
import pickle
import hashlib
import uuid
import os
import io
from datetime import datetime
import sqlite3
import matplotlib.pyplot as plt
import pyaudio
import struct
import matplotlib
import json
import asyncio
import websockets
import threading
from collections import deque
matplotlib.use('TkAgg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max-limit
app.config['DATABASE'] = 'encrypted_files.db'
WEBSOCKET_CLIENTS = set()

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class VoiceAnalyzer:
    def __init__(self):
        self.FRAMES = 10248
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.Fs = 44100
        self.required_samples = 5
        self.samples = []
        self.coherence_threshold = 0.7
        self.similarity_threshold = 0.85
        
        # Inicializar PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.Fs,
            input=True,
            output=True,
            frames_per_buffer=self.FRAMES,
            stream_callback=self.audio_callback
        )
        
        # Configurar visualización
        self.fig, (self.ax, self.ax1) = plt.subplots(2)
        self.x_audio = np.arange(0, self.FRAMES, 1)
        self.x_fft = np.linspace(0, self.Fs, self.FRAMES)
        self.line, = self.ax.plot(self.x_audio, np.random.rand(self.FRAMES), 'r')
        self.line_fft, = self.ax1.semilogx(self.x_fft, np.random.rand(self.FRAMES), 'b')
        
        # Configurar ejes
        self.ax.set_ylim(-32500, 32500)
        self.ax.set_xlim(0, self.FRAMES)
        self.ax1.set_xlim(1, 5000)
        
        self.fig.show()

    def audio_callback(self, in_data, frame_count, time_info, status):
        try:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            fft_data = self.compute_fft(audio_data)
            coherence_score = self.analyze_speech_coherence(audio_data)

            # Actualizar gráficos
            self.update_plots(audio_data, fft_data)
            
            # Enviar datos a clientes WebSocket
            data_to_send = {
                'fft_data': fft_data.tolist()[:100],  # Primeras 100 frecuencias
                'coherence_score': float(coherence_score),
                'is_valid_sample': coherence_score > self.coherence_threshold
            }
            
            asyncio.run(self.broadcast_data(data_to_send))

            # Si la muestra es válida, la guardamos
            if coherence_score > self.coherence_threshold:
                if len(self.samples) < self.required_samples:
                    self.samples.append(fft_data)
            
            return (in_data, pyaudio.paContinue)
        except Exception as e:
            print(f"Error en audio_callback: {str(e)}")
            return (in_data, pyaudio.paContinue)

    def compute_fft(self, audio_data):
        """Calcula la FFT normalizada"""
        windowed_data = signal.windows.hann(len(audio_data)) * audio_data
        fft_data = np.abs(fft(windowed_data))
        return fft_data / np.max(fft_data)

    def analyze_speech_coherence(self, audio_data):
        """Analiza la coherencia del habla usando múltiples métricas"""
        # Energía de la señal
        energy = np.mean(audio_data ** 2)
        
        # Tasa de cruces por cero
        zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
        
        # Análisis espectral
        freqs, times, Sxx = signal.spectrogram(audio_data, fs=self.Fs)
        spectral_flatness = np.exp(np.mean(np.log(Sxx + 1e-10))) / np.mean(Sxx + 1e-10)
        
        # Combinar métricas
        coherence_score = (
            0.4 * (energy > 1000) +  # Peso para energía
            0.3 * (100 < zero_crossings < 1000) +  # Peso para cruces por cero
            0.3 * (spectral_flatness < 0.5)  # Peso para planitud espectral
        )
        
        return float(coherence_score)

    def update_plots(self, audio_data, fft_data):
        """Actualiza las gráficas en tiempo real"""
        self.line.set_ydata(audio_data)
        self.line_fft.set_ydata(fft_data)
        self.ax1.set_ylim(0, np.max(fft_data) + 10)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    async def broadcast_data(self, data):
        """Envía datos a todos los clientes WebSocket"""
        message = json.dumps(data)
        websockets_to_remove = set()
        
        for websocket in WEBSOCKET_CLIENTS:
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                websockets_to_remove.add(websocket)
            except Exception as e:
                print(f"Error broadcasting: {str(e)}")
                websockets_to_remove.add(websocket)
        
        # Eliminar conexiones cerradas
        WEBSOCKET_CLIENTS.difference_update(websockets_to_remove)

    def analyze_voice(self):
        """Analiza la voz y espera hasta tener suficientes muestras válidas"""
        self.samples = []
        frames = []
        print("Iniciando grabación...")
        
        while len(self.samples) < self.required_samples:
            try:
                data = self.stream.read(self.FRAMES)
                dataInt = struct.unpack(str(self.FRAMES) + 'h', data)
                frames.extend(dataInt)
                
                audio_data = np.array(dataInt)
                fft_data = self.compute_fft(audio_data)
                coherence_score = self.analyze_speech_coherence(audio_data)
                
                self.update_plots(audio_data, fft_data)
                
                if coherence_score > self.coherence_threshold:
                    print(f"Muestra válida recolectada ({len(self.samples) + 1}/5)")
                    self.samples.append(fft_data)
                
            except Exception as e:
                print(f"Error durante la grabación: {str(e)}")
        
        print("Verificando consistencia de las muestras...")
        if not self.verify_samples():
            raise Exception("Las muestras de voz no son consistentes. Por favor, repita la clave.")
            
        # Retornar el promedio de las muestras
        average_fft = np.mean(self.samples, axis=0)
        print("Análisis de voz completado exitosamente")
        return np.array(frames, dtype=np.float32), average_fft

    def verify_samples(self):
        """Verifica que las muestras sean similares entre sí"""
        similarities = []
        for i in range(len(self.samples)):
            for j in range(i + 1, len(self.samples)):
                corr, _ = pearsonr(self.samples[i], self.samples[j])
                similarities.append(corr)
        
        avg_similarity = np.mean(similarities)
        print(f"Similitud promedio entre muestras: {avg_similarity:.2f}")
        return avg_similarity > self.similarity_threshold

    def close(self):
        """Cierra los recursos"""
        print("Cerrando recursos de audio...")
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        plt.close(self.fig)

class VoiceEncryption:
    def __init__(self, matrix_size=16):
        self.matrix_size = matrix_size
        self.voice_analyzer = VoiceAnalyzer()
    
    def generate_key_matrix(self, voice_data=None):
        """Genera matriz clave a partir de FFT"""
        try:
            print("Generando matriz clave...")
            audio_data, fft_data = self.voice_analyzer.analyze_voice()
            magnitudes = np.abs(fft_data[:self.matrix_size**2])
            
            # Normalizar y crear matriz
            normalized = (magnitudes - np.min(magnitudes)) / (np.max(magnitudes) - np.min(magnitudes))
            matrix = normalized.reshape(self.matrix_size, self.matrix_size)
            
            # Asegurar que sea invertible
            while np.linalg.matrix_rank(matrix) < self.matrix_size:
                matrix += np.eye(self.matrix_size) * 0.01
                
            return matrix, fft_data.tolist()
            
        except Exception as e:
            print(f"Error en generate_key_matrix: {str(e)}")
            raise

    def encrypt_file(self, file_data, key_matrix):
        """Cifra un archivo usando la matriz clave"""
        data_array = np.frombuffer(file_data, dtype=np.uint8)
        padding_size = (self.matrix_size - (len(data_array) % self.matrix_size)) % self.matrix_size
        
        if padding_size:
            data_array = np.pad(data_array, (0, padding_size))
            
        blocks = data_array.reshape(-1, self.matrix_size)
        encrypted_blocks = np.dot(blocks, key_matrix)
        
        return {
            'data': encrypted_blocks,
            'original_size': len(file_data),
            'padding': padding_size
        }

    def decrypt_file(self, encrypted_data, key_matrix):
        """Descifra un archivo usando la matriz clave"""
        try:
            inverse_matrix = linalg.inv(key_matrix)
            decrypted_blocks = np.dot(encrypted_data['data'], inverse_matrix)
            decrypted_bytes = np.round(decrypted_blocks).astype(np.uint8)
            decrypted_bytes = decrypted_bytes.flatten()[:encrypted_data['original_size']]
            return bytes(decrypted_bytes)
        except np.linalg.LinAlgError:
            raise Exception("Error: La matriz clave no es invertible")

async def websocket_handler(websocket, path):
    """Maneja las conexiones WebSocket"""
    print(f"Nueva conexión WebSocket establecida")
    WEBSOCKET_CLIENTS.add(websocket)
    try:
        while True:
            await websocket.recv()  # Mantener conexión abierta
    except websockets.exceptions.ConnectionClosed:
        print("Conexión WebSocket cerrada")
    finally:
        WEBSOCKET_CLIENTS.remove(websocket)

def run_websocket_server():
    """Ejecuta el servidor WebSocket"""
    print("Iniciando servidor WebSocket...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(websocket_handler, "localhost", 8765)
    loop.run_until_complete(start_server)
    loop.run_forever()

def init_db():
    """Inicializa la base de datos"""
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS encrypted_files
        (id TEXT PRIMARY KEY,
         filename TEXT NOT NULL,
         key_hash TEXT NOT NULL,
         upload_date DATETIME NOT NULL,
         original_filename TEXT NOT NULL,
         nickname TEXT NOT NULL,
         voice_data BLOB)
    ''')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encrypt', methods=['POST'])
def encrypt():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    nickname = request.form.get('nickname', '')
    
    if not nickname:
        return jsonify({'error': 'Nickname is required'}), 400

    try:
        print(f"Iniciando proceso de cifrado para {nickname}")
        encryptor = VoiceEncryption()
        
        key_matrix, fft_data = encryptor.generate_key_matrix()
        key_hash = hashlib.sha256(key_matrix.tobytes()).hexdigest()
        file_id = str(uuid.uuid4())
        
        print("Cifrando archivo...")
        file_data = file.read()
        encrypted_data = encryptor.encrypt_file(file_data, key_matrix)
        
        encrypted_filename = f"{file_id}.enc"
        encrypted_path = os.path.join(app.config['UPLOAD_FOLDER'], encrypted_filename)
        
        print("Guardando archivo cifrado...")
        with open(encrypted_path, 'wb') as f:
            pickle.dump(encrypted_data, f)
        
        print("Actualizando base de datos...")
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()
        c.execute('''
            INSERT INTO encrypted_files 
            (id, filename, key_hash, upload_date, original_filename, nickname, voice_data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (file_id, encrypted_filename, key_hash, datetime.now(), 
              file.filename, nickname, pickle.dumps(fft_data)))
        conn.commit()
        conn.close()
        
        print("Cifrado completado exitosamente")
        return jsonify({
            'message': 'File encrypted successfully',
            'file_id': file_id,
            'fft_data': fft_data
        })
        
    except Exception as e:
        print(f"Error durante el cifrado: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        if 'encryptor' in locals():
            encryptor.voice_analyzer.close()

@app.route('/decrypt/<file_id>', methods=['POST'])
def decrypt(file_id):
    try:
        print(f"Iniciando proceso de descifrado para archivo {file_id}")
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()
        c.execute('SELECT filename, original_filename, key_hash FROM encrypted_files WHERE id = ?', 
                 (file_id,))
        result = c.fetchone()
        conn.close()
        
        if not result:
            return jsonify({'error': 'File not found'}), 404
            
        encrypted_filename, original_filename, stored_key_hash = result
        
        print("Generando matriz clave...")
        encryptor = VoiceEncryption()
        key_matrix, _ = encryptor.generate_key_matrix()
        
        print("Verificando hash de voz...")
        new_key_hash = hashlib.sha256(key_matrix.tobytes()).hexdigest()
        if new_key_hash != stored_key_hash:
            return jsonify({'error': 'Invalid voice key - La clave de voz no coincide'}), 401
        
        print("Cargando archivo cifrado...")
        encrypted_path = os.path.join(app.config['UPLOAD_FOLDER'], encrypted_filename)
        with open(encrypted_path, 'rb') as f:
            encrypted_data = pickle.load(f)
        
        print("Descifrando archivo...")
        decrypted_data = encryptor.decrypt_file(encrypted_data, key_matrix)
        
        print("Descifrado completado exitosamente")
        return send_file(
            io.BytesIO(decrypted_data),
            as_attachment=True,
            download_name=original_filename
        )
        
    except Exception as e:
        print(f"Error durante el descifrado: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        if 'encryptor' in locals():
            encryptor.voice_analyzer.close()

@app.route('/status/recording', methods=['GET'])
def get_recording_status():
    """Endpoint para obtener el estado actual de la grabación"""
    try:
        voice_analyzer = VoiceAnalyzer()
        status = {
            'samples_collected': len(voice_analyzer.samples),
            'required_samples': voice_analyzer.required_samples,
            'is_complete': len(voice_analyzer.samples) >= voice_analyzer.required_samples
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if 'voice_analyzer' in locals():
            voice_analyzer.close()

@app.route('/verify_voice', methods=['POST'])
def verify_voice():
    """Endpoint para verificar si una muestra de voz es válida"""
    try:
        voice_analyzer = VoiceAnalyzer()
        audio_data, fft_data = voice_analyzer.analyze_voice()
        coherence_score = voice_analyzer.analyze_speech_coherence(audio_data)
        
        return jsonify({
            'is_valid': coherence_score > voice_analyzer.coherence_threshold,
            'coherence_score': float(coherence_score),
            'samples_collected': len(voice_analyzer.samples),
            'required_samples': voice_analyzer.required_samples
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if 'voice_analyzer' in locals():
            voice_analyzer.close()

def cleanup_old_files():
    """Limpia archivos cifrados antiguos (más de 24 horas)"""
    try:
        print("Iniciando limpieza de archivos antiguos...")
        current_time = datetime.now()
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()
        
        # Obtener archivos antiguos
        c.execute('''
            SELECT filename FROM encrypted_files 
            WHERE julianday('now') - julianday(upload_date) > 1
        ''')
        old_files = c.fetchall()
        
        # Eliminar archivos
        for file_name, in old_files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Archivo eliminado: {file_name}")
        
        # Limpiar registros de la base de datos
        c.execute('''
            DELETE FROM encrypted_files 
            WHERE julianday('now') - julianday(upload_date) > 1
        ''')
        
        conn.commit()
        conn.close()
        print("Limpieza completada")
        
    except Exception as e:
        print(f"Error durante la limpieza: {str(e)}")

def start_cleanup_scheduler():
    """Inicia el programador de limpieza"""
    def run_cleanup():
        while True:
            cleanup_old_files()
            # Esperar 12 horas
            time.sleep(43200)
    
    cleanup_thread = threading.Thread(target=run_cleanup)
    cleanup_thread.daemon = True
    cleanup_thread.start()

if __name__ == '__main__':
    print("Iniciando sistema de cifrado por voz...")
    init_db()
    
    # Iniciar servidor WebSocket en un thread separado
    print("Iniciando servidor WebSocket...")
    websocket_thread = threading.Thread(target=run_websocket_server)
    websocket_thread.daemon = True
    websocket_thread.start()
    
    # Iniciar el programador de limpieza
    start_cleanup_scheduler()
    
    # Iniciar servidor Flask
    print("Iniciando servidor web...")
    app.run(debug=True, host='0.0.0.0', port=5000)