"""ROS2 node for HMM+MFCC voice recognition.

Publications
------------
/voice_command  (std_msgs/String)
    Recognised word published after each utterance.

/mission  (std_msgs/String)
    JSON-serialised mission dict when a grammar rule matches.

Subscriptions (mock_mode only)
-------------------------------
/mock_voice  (std_msgs/String)
    Inject words directly for testing without a microphone.

Parameters
----------
See config/voice_params.yaml for the full list.

Voice command py
"""

from __future__ import annotations

import json
import os
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from voice_control.command_parser import parse_command
from voice_control.hmm_recognizer import HMMRecognizer


class VoiceNode(Node):
    """HMM-based voice recognition bridged to ROS2."""

    def __init__(self) -> None:
        super().__init__('voice_node')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('models_dir', '')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('word_buffer_size', 5)
        self.declare_parameter('mock_mode', True)

        p = self.get_parameter
        self._sample_rate: int = (
            p('sample_rate').get_parameter_value().integer_value
        )
        self._word_buffer_size: int = (
            p('word_buffer_size').get_parameter_value().integer_value
        )
        self._mock_mode: bool = (
            p('mock_mode').get_parameter_value().bool_value
        )

        models_dir: str = p('models_dir').get_parameter_value().string_value
        if not models_dir:
            models_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'models',
            )

        # ── State ─────────────────────────────────────────────────────
        self._word_buffer: list[str] = []

        # ── ROS interfaces ────────────────────────────────────────────
        self._voice_pub   = self.create_publisher(String, '/voice_command', 10)
        self._mission_pub = self.create_publisher(String, '/mission', 10)

        # ── Recognizer ────────────────────────────────────────────────
        self._recognizer = HMMRecognizer(models_dir=models_dir)
        if self._recognizer.load():
            self.get_logger().info(
                f'HMM models loaded: {self._recognizer.vocabulary}'
            )
        else:
            self.get_logger().warn(
                f'HMM models not found in {models_dir}. '
                'Recognition unavailable until models are present.'
            )

        # ── Mode ──────────────────────────────────────────────────────
        if self._mock_mode:
            self._setup_mock_mode()
        else:
            self._setup_real_mode()

        self.get_logger().info(
            f'VoiceNode ready — mock_mode={self._mock_mode}, '
            f'models_dir={models_dir}'
        )

    # ── Mock mode ───────────────────────────────

    def _setup_mock_mode(self) -> None:
        self._mock_sub = self.create_subscription(
            String, '/mock_voice', self._on_mock_voice, 10
        )
        self.get_logger().info(
            'VoiceNode in mock mode — publish words to /mock_voice to test'
        )

    def _on_mock_voice(self, msg: String) -> None:
        word = msg.data.strip().lower()
        if word:
            self.get_logger().info(f'Mock voice word: "{word}"')
            self._handle_word(word)

    # ── Real microphone mode ───────────────────────────────────────────

    def _setup_real_mode(self) -> None:
        self._stop_event  = threading.Event()
        self._audio_thread = threading.Thread(
            target=self._audio_loop, name='voice_audio_loop', daemon=True
        )
        self._audio_thread.start()
        self.get_logger().info('VoiceNode audio thread started')

    def _audio_loop(self) -> None:
        """Capture audio from microphone, segment utterances, recognise."""
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            self.get_logger().error(f'Audio import failed: {exc}')
            return

        fs             = self._sample_rate
        buf: list      = []
        silence_frames = 0
        # Adaptive RMS threshold: raised after each utterance to reduce false triggers
        SILENCE_THRESH = 0.015
        MAX_SILENCE_S  = 0.7   # end utterance after 0.7 s of silence
        MIN_SPEECH_S   = 0.15  # discard buffers shorter than this

        def _rms(chunk):
            return float(np.sqrt(np.mean(chunk ** 2)))

        self.get_logger().info('Opening microphone stream…')
        try:
            with sd.InputStream(samplerate=fs, channels=1, dtype='float32',
                                blocksize=1024) as stream:
                self.get_logger().info('Microphone open — listening')
                in_speech = False
                while not self._stop_event.is_set():
                    chunk, _ = stream.read(1024)
                    chunk    = chunk[:, 0]
                    rms      = _rms(chunk)

                    if rms > SILENCE_THRESH:
                        in_speech      = True
                        silence_frames = 0
                        buf.append(chunk)
                    elif in_speech:
                        buf.append(chunk)
                        silence_frames += 1
                        max_sil = int(MAX_SILENCE_S * fs / 1024)
                        if silence_frames >= max_sil:
                            signal = np.concatenate(buf)
                            if len(signal) >= int(MIN_SPEECH_S * fs):
                                self._process_utterance(signal)
                            buf            = []
                            in_speech      = False
                            silence_frames = 0
        except Exception as exc:
            self.get_logger().error(f'Audio loop error: {exc}')

    def _process_utterance(self, signal) -> None:
        """Apply VAD endpoint trimming then run HMM recognition."""
        try:
            from voice_control.hmm_utils import detect_endpoints
            start, end = detect_endpoints(signal, fs=self._sample_rate)
            segment    = signal[start:end]
            if len(segment) < int(0.1 * self._sample_rate):
                self.get_logger().debug('Utterance too short after VAD — skipping')
                return
        except Exception:
            segment = signal

        word = self._recognizer.recognize(segment, fs=self._sample_rate)
        if word:
            self.get_logger().info(f'Recognised: "{word}"')
            self._handle_word(word)

    # ── Shared word handling ───────────────────────────────────────────

    def _handle_word(self, word: str) -> None:
        msg = String()
        msg.data = word
        self._voice_pub.publish(msg)

        self._word_buffer.append(word)
        if len(self._word_buffer) > self._word_buffer_size:
            self._word_buffer.pop(0)

        command = parse_command(self._word_buffer)
        if command is not None:
            self.get_logger().info(f'Command parsed: {command}')
            m = String()
            m.data = json.dumps(command)
            self._mission_pub.publish(m)
            self._word_buffer.clear()

    # ── Lifecycle ─────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        if hasattr(self, '_stop_event'):
            self._stop_event.set()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VoiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
