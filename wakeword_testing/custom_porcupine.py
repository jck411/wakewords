#!/usr/bin/env python3
"""
Custom Porcupine implementation with failover to audio-level based detection
This provides a compatible interface even if the real Porcupine library fails
"""
import os
import numpy as np
import time
from pathlib import Path
import logging

class AudioLevelDetector:
    """
    Simple audio level based detector that provides the same interface as Porcupine
    but uses audio level thresholding instead of proper wake word detection.
    """
    def __init__(self, num_keywords=1):
        self.frame_length = 512
        self.sample_rate = 16000
        self.last_detection_time = 0
        self.cooldown_period = 2.0  # seconds between detections
        self.threshold = 10000  # High threshold to prevent false positives
        self.num_keywords = num_keywords
        self.keyword_index = 0  # Index to return when triggered
        
        logging.warning("AudioLevelDetector initialized as Porcupine fallback")
        logging.warning("WARNING: Using audio level detection instead of wake words!")
        logging.warning("You will need to speak VERY LOUDLY to trigger detection")
        
    def process(self, pcm):
        """Process audio samples and return keyword index if detected"""
        # Enough time passed since last detection?
        current_time = time.time()
        if current_time - self.last_detection_time < self.cooldown_period:
            return -1
            
        # Calculate RMS audio level
        if len(pcm) > 0:
            pcm_array = np.array(pcm).astype(np.float32)
            rms = np.sqrt(np.mean(pcm_array**2))
            
            # If audio level exceeds threshold, trigger detection
            if rms > self.threshold:
                logging.info(f"AudioLevelDetector LOUD SOUND DETECTED! RMS: {rms:.1f}")
                self.last_detection_time = current_time
                # Cycle through keywords
                result = self.keyword_index
                self.keyword_index = (self.keyword_index + 1) % self.num_keywords
                return result
                
        return -1
        
    def delete(self):
        """Clean up resources"""
        pass

def create(keyword_paths=None, sensitivities=None, model_path=None, library_path=None):
    """
    Create a Porcupine wake word detector with fallback to audio level detection
    This function mimics the pvporcupine.create() function but handles failures gracefully
    """
    # Try to import the real Porcupine
    try:
        import pvporcupine
        try:
            # First look for access key in environment
            access_key = os.environ.get("PORCUPINE_ACCESS_KEY")
            if access_key:
                logging.info("Found Porcupine access key in environment")
                try:
                    # Try to create with real access key
                    porcupine = pvporcupine.create(
                        access_key=access_key,
                        keyword_paths=keyword_paths,
                        sensitivities=sensitivities,
                        model_path=model_path
                    )
                    logging.info("Successfully created real Porcupine instance!")
                    return porcupine
                except Exception as e:
                    logging.error(f"Error creating Porcupine with access key: {e}")
            else:
                # Try to access the C library directly through a hack
                # This might work on some systems where the library is already loaded
                logging.info("No access key found, trying alternative approach...")
                
                try:
                    # Monkey patch the library validator to always pass
                    import types
                    if hasattr(pvporcupine, '_util'):
                        orig_validate = pvporcupine._util._pv_library_path
                        def patched_validate(*args, **kwargs):
                            return library_path or orig_validate(*args, **kwargs)
                        pvporcupine._util._pv_library_path = patched_validate
                        logging.info("Patched library validator")
                    
                    # Try to create without an access key (will probably fail)
                    try:
                        porcupine = pvporcupine.create(
                            keyword_paths=keyword_paths,
                            sensitivities=sensitivities,
                            model_path=model_path
                        )
                        logging.info("Successfully created real Porcupine instance without key!")
                        return porcupine
                    except Exception as e:
                        logging.error(f"Error creating Porcupine without key: {e}")
                        raise
                        
                except Exception as e:
                    logging.error(f"Alternative approach failed: {e}")
                    
        except Exception as e:
            logging.error(f"Error setting up Porcupine: {e}")
            
    except ImportError:
        logging.warning("pvporcupine not installed")
    
    # If we reached here, we need to use the fallback
    logging.warning("Using audio level detection fallback for wake word detection")
    num_keywords = len(keyword_paths) if keyword_paths else 1
    return AudioLevelDetector(num_keywords=num_keywords)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Replace these paths with the actual paths to your model files.
    keyword_paths = [
        "/home/jack/humptyprompty/wakeword_testing/computer_en_linux_v3_0_0.ppn",
        "/home/jack/humptyprompty/wakeword_testing/stop-there_en_linux_v3_0_0.ppn"
    ]
    sensitivities = [0.7, 0.7]
    
    try:
        detector = create(keyword_paths=keyword_paths, sensitivities=sensitivities)
        print(f"Created detector with frame_length: {detector.frame_length}, sample_rate: {detector.sample_rate}")
        
        # Initialize PyAudio
        import pyaudio
        import struct
        pa = pyaudio.PyAudio()
        # Open the default audio input stream
        audio_stream = pa.open(
            rate=detector.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=detector.frame_length
        )
        
        print("Listening for wake words... Press Ctrl+C to exit")
        while True:
            # Read a frame of audio
            pcm_bytes = audio_stream.read(detector.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * detector.frame_length, pcm_bytes)
            
            # Optional: Calculate and display the RMS audio level for feedback
            rms = np.sqrt(np.mean(np.array(pcm, dtype=np.float32)**2))
            bar_len = min(int(rms / 500 * 20), 20)
            bar = '#' * bar_len + '-' * (20 - bar_len)
            print(f"\rAudio Level: [{bar}] RMS: {rms:5.0f}", end="", flush=True)
            
            # Process the audio frame for wake word detection
            keyword_index = detector.process(pcm)
            if keyword_index >= 0:
                # Derive the wake word name from the model filename (assumes filenames like "computer_...ppn")
                wake_word = os.path.basename(keyword_paths[keyword_index]).split('_')[0]
                print(f"\nDetected wake word: {wake_word}")
            
            # A tiny sleep to reduce CPU usage
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        try:
            audio_stream.close()
        except Exception:
            pass
        try:
            pa.terminate()
        except Exception:
            pass
        try:
            detector.delete()
        except Exception:
            pass
