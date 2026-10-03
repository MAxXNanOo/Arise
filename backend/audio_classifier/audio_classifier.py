import queue
import multiprocessing as mp

import numpy as np


def yamnet_worker(
    input_queue,
    output_queue,
    sample_rate,
    chunk_duration
):

    try:

        import sounddevice as sd
        import tensorflow as tf
        import tensorflow_hub as hub

        print("[Audio] Loading YAMNet...")

        yamnet = hub.load(
            "https://tfhub.dev/google/yamnet/1"
        )

        class_map_path = tf.keras.utils.get_file(
            "yamnet_class_map.csv",
            "https://raw.githubusercontent.com/"
            "tensorflow/models/master/"
            "research/audioset/yamnet/"
            "yamnet_class_map.csv"
        )

        class_names = []

        with open(
            class_map_path,
            "r",
            encoding="utf-8"
        ) as f:

            next(f)

            for line in f:

                parts = line.strip().split(
                    ",",
                    2
                )

                if len(parts) >= 3:
                    class_names.append(
                        parts[2].strip()
                    )

        print("[Audio] YAMNet ready")

        block_size = int(
            sample_rate *
            chunk_duration
        )

        while True:

            try:

                audio = sd.rec(
                    block_size,
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32"
                )

                sd.wait()

                audio = audio.reshape(-1)

                if len(audio) == 0:

                    output_queue.put({
                        "class_name": "No Audio",
                        "confidence": 0.0
                    })

                    continue

                scores, embeddings, spectrogram = (
                    yamnet(
                        tf.convert_to_tensor(
                            audio,
                            dtype=tf.float32
                        )
                    )
                )

                scores_np = scores.numpy()

                mean_scores = np.mean(
                    scores_np,
                    axis=0
                )

                class_id = int(
                    np.argmax(mean_scores)
                )

                confidence = float(
                    mean_scores[class_id]
                )

                if class_id < len(
                    class_names
                ):

                    class_name = (
                        class_names[class_id]
                    )

                else:

                    class_name = str(
                        class_id
                    )

                output_queue.put({

                    "class_name":
                        class_name,

                    "confidence":
                        confidence
                })

            except Exception as e:

                output_queue.put({
                    "class_name": "No Audio",
                    "confidence": 0.0
                })

                print(
                    f"[Audio] Error: {e}"
                )

    except Exception as e:

        print(
            f"[Audio] Worker startup error: {e}"
        )

        output_queue.put({
            "class_name": "No Audio",
            "confidence": 0.0
        })


class AudioController:

    def __init__(self, config):

        audio_config = config.get(
            "audio",
            {}
        )

        self.sample_rate = int(
            audio_config.get(
                "sample_rate",
                16000
            )
        )

        self.chunk_duration = float(
            audio_config.get(
                "chunk_duration_sec",
                0.975
            )
        )

        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()

        self.process = None

        self.latest = {
            "class_name": "No Audio",
            "confidence": 0.0
        }

    def start(self):

        print(
            "[Audio] Starting process..."
        )

        self.process = mp.Process(
            target=yamnet_worker,
            args=(
                self.input_queue,
                self.output_queue,
                self.sample_rate,
                self.chunk_duration
            ),
            daemon=True
        )

        self.process.start()

    def get_latest(self):

        try:

            while True:

                self.latest = (
                    self.output_queue
                    .get_nowait()
                )

        except queue.Empty:
            pass

        except Exception:
            pass

        return self.latest

    def stop(self):

        if (
            self.process is not None
            and self.process.is_alive()
        ):

            self.process.terminate()

            self.process.join(
                timeout=2
            )