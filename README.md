# src/models/user.py
from dataclasses import dataclass
from typing import Optional
import uuid

@dataclass
class User:
    user_id: str
    user_name: str
    email: Optional[str] = None
    created_at: Optional[str] = None
    
    def __post_init__(self):
        if not self.user_id:
            self.user_id = str(uuid.uuid4())
    
    def request_bgm(self, preferences):
        """BGMをリクエストする"""
        return f"User {self.user_name} requested BGM with preferences: {preferences}"
    
    def send_feedback(self, bgm, rating: int, comment: str = ""):
        """フィードバックを送る"""
        return {
            "user_id": self.user_id,
            "bgm_id": bgm.bgm_id,
            "rating": rating,
            "comment": comment
        }

# src/models/user_preference.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class UserPreference:
    mood: str  # "relaxed", "energetic", "focus", "creative"
    genre: str  # "ambient", "electronic", "classical", "jazz"
    tempo: str  # "slow", "medium", "fast"
    instruments: List[str]  # ["piano", "guitar", "synth", "drums"]
    duration: int  # 再生時間（秒）
    key_signature: Optional[str] = "C"  # 調性
    time_signature: Optional[str] = "4/4"  # 拍子
    
    def validate(self) -> bool:
        """設定の妥当性をチェック"""
        valid_moods = ["relaxed", "energetic", "focus", "creative"]
        valid_genres = ["ambient", "electronic", "classical", "jazz"]
        valid_tempos = ["slow", "medium", "fast"]
        
        return (
            self.mood in valid_moods and
            self.genre in valid_genres and
            self.tempo in valid_tempos and
            self.duration > 0 and
            len(self.instruments) > 0
        )

# src/models/bgm.py
from dataclasses import dataclass
from typing import Optional
import uuid
from datetime import datetime

@dataclass
class BGM:
    bgm_id: str
    title: str
    file_path: str
    related_preference: 'UserPreference'
    created_at: Optional[datetime] = None
    file_size: Optional[int] = None
    duration: Optional[float] = None
    
    def __post_init__(self):
        if not self.bgm_id:
            self.bgm_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now()
    
    def get_metadata(self) -> dict:
        """BGMのメタデータを取得"""
        return {
            "id": self.bgm_id,
            "title": self.title,
            "duration": self.duration,
            "file_size": self.file_size,
            "created_at": self.created_at.isoformat(),
            "preferences": {
                "mood": self.related_preference.mood,
                "genre": self.related_preference.genre,
                "tempo": self.related_preference.tempo,
                "instruments": self.related_preference.instruments
            }
        }

# src/models/audio_components.py
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class Note:
    pitch: str  # "C4", "D#3", etc.
    duration: float  # 音符の長さ（秒）
    velocity: int  # 音の強さ（0-127）
    start_time: float = 0.0
    
    def to_frequency(self) -> float:
        """音高を周波数に変換"""
        # A4 = 440Hz を基準とした計算
        note_map = {
            'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4,
            'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2
        }
        
        note_name = self.pitch[:-1]
        octave = int(self.pitch[-1])
        
        semitones = note_map[note_name] + (octave - 4) * 12
        return 440 * (2 ** (semitones / 12))

@dataclass
class Chord:
    root: str  # ルート音
    quality: str  # "major", "minor", "diminished", "augmented"
    notes: List[Note]
    duration: float = 4.0
    
    @classmethod
    def from_root_and_quality(cls, root: str, quality: str, octave: int = 4):
        """ルート音と品質からコードを生成"""
        intervals = {
            "major": [0, 4, 7],
            "minor": [0, 3, 7],
            "diminished": [0, 3, 6],
            "augmented": [0, 4, 8]
        }
        
        notes = []
        for interval in intervals[quality]:
            # 簡単な実装: 半音階で音程を計算
            notes.append(Note(
                pitch=f"{root}{octave}",
                duration=4.0,
                velocity=80
            ))
        
        return cls(root=root, quality=quality, notes=notes)

@dataclass
class Instrument:
    name: str
    type: str  # "acoustic", "electric", "synthesizer"
    parameters: Dict[str, Any]
    
    def get_sound_parameters(self) -> Dict[str, Any]:
        """楽器の音響パラメータを取得"""
        base_params = {
            "attack": 0.1,
            "decay": 0.3,
            "sustain": 0.7,
            "release": 0.5,
            "volume": 1.0
        }
        base_params.update(self.parameters)
        return base_params

@dataclass
class AudioClip:
    audio_data: np.ndarray
    sample_rate: int
    format: str = "wav"
    
    @property
    def duration(self) -> float:
        """音声の長さを秒で返す"""
        return len(self.audio_data) / self.sample_rate
    
    def normalize(self) -> 'AudioClip':
        """音量を正規化"""
        max_val = np.max(np.abs(self.audio_data))
        if max_val > 0:
            normalized_data = self.audio_data / max_val
        else:
            normalized_data = self.audio_data
        
        return AudioClip(
            audio_data=normalized_data,
            sample_rate=self.sample_rate,
            format=self.format
        )

# src/services/music_theory_engine.py
import random
from typing import List, Dict, Tuple
from ..models.audio_components import Note, Chord
from ..models.user_preference import UserPreference

class MusicTheoryEngine:
    def __init__(self):
        self.scale_patterns = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "pentatonic": [0, 2, 4, 7, 9],
            "blues": [0, 3, 5, 6, 7, 10]
        }
        
        self.chord_progressions = {
            "major": ["I", "V", "vi", "IV"],
            "minor": ["i", "VII", "VI", "VII"],
            "jazz": ["IIM7", "V7", "IM7", "VIM7"]
        }
    
    def generate_melody(self, preference: UserPreference) -> List[Note]:
        """ユーザー設定に基づいてメロディを生成"""
        scale = self._select_scale(preference.genre)
        tempo_bpm = self._determine_tempo(preference.tempo)
        
        melody = []
        current_time = 0.0
        note_duration = 60.0 / tempo_bpm  # 四分音符の長さ
        
        # メロディの長さを決定
        num_notes = int(preference.duration / note_duration)
        
        for i in range(num_notes):
            # スケールからランダムに音を選択
            scale_degree = random.choice(scale)
            octave = random.choice([3, 4, 5])
            
            # 音高を計算
            pitch = self._scale_degree_to_pitch(scale_degree, preference.key_signature, octave)
            
            # 音価をランダムに決定
            duration_multiplier = random.choice([0.5, 1.0, 1.5, 2.0])
            duration = note_duration * duration_multiplier
            
            # 音の強さを決定
            velocity = self._calculate_velocity(preference.mood, i, num_notes)
            
            note = Note(
                pitch=pitch,
                duration=duration,
                velocity=velocity,
                start_time=current_time
            )
            
            melody.append(note)
            current_time += duration
        
        return melody
    
    def generate_chords(self, melody: List[Note]) -> List[Chord]:
        """メロディに基づいてコード進行を生成"""
        chords = []
        
        # 簡単なコード進行パターンを適用
        chord_pattern = ["C", "Am", "F", "G"]  # vi-IV-I-V
        
        measures = len(melody) // 4  # 4つの音符で1小節と仮定
        
        for i in range(measures):
            chord_root = chord_pattern[i % len(chord_pattern)]
            chord = Chord.from_root_and_quality(chord_root, "major")
            chords.append(chord)
        
        return chords
    
    def determine_tempo(self, tempo_preference: str) -> int:
        """テンポ設定からBPMを決定"""
        tempo_map = {
            "slow": random.randint(60, 80),
            "medium": random.randint(90, 120),
            "fast": random.randint(130, 160)
        }
        return tempo_map.get(tempo_preference, 120)
    
    def _select_scale(self, genre: str) -> List[int]:
        """ジャンルに基づいてスケールを選択"""
        genre_scales = {
            "classical": "major",
            "ambient": "minor",
            "jazz": "blues",
            "electronic": "pentatonic"
        }
        
        scale_type = genre_scales.get(genre, "major")
        return self.scale_patterns[scale_type]
    
    def _determine_tempo(self, tempo_preference: str) -> int:
        """テンポ設定からBPMを決定"""
        return self.determine_tempo(tempo_preference)
    
    def _scale_degree_to_pitch(self, scale_degree: int, key: str, octave: int) -> str:
        """スケール度数を音高に変換"""
        chromatic_scale = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        
        # キーのオフセットを計算
        key_offset = chromatic_scale.index(key) if key in chromatic_scale else 0
        
        # 絶対音高を計算
        pitch_class = (key_offset + scale_degree) % 12
        pitch_name = chromatic_scale[pitch_class]
        
        return f"{pitch_name}{octave}"
    
    def _calculate_velocity(self, mood: str, position: int, total_notes: int) -> int:
        """ムードと位置に基づいて音の強さを計算"""
        base_velocity = {
            "relaxed": 60,
            "energetic": 100,
            "focus": 80,
            "creative": 90
        }.get(mood, 80)
        
        # 曲の展開に応じて強弱を調整
        progress = position / total_notes
        if progress < 0.3:  # 序盤
            return base_velocity - 10
        elif progress > 0.7:  # 終盤
            return base_velocity + 10
        else:  # 中盤
            return base_velocity

# src/services/sound_library.py
import numpy as np
from typing import List, Dict
from ..models.audio_components import Instrument, AudioClip, Note

class SoundLibrary:
    def __init__(self):
        self.instruments = self._initialize_instruments()
    
    def _initialize_instruments(self) -> Dict[str, Instrument]:
        """利用可能な楽器を初期化"""
        instruments = {
            "piano": Instrument(
                name="piano",
                type="acoustic",
                parameters={
                    "attack": 0.01,
                    "decay": 0.3,
                    "sustain": 0.7,
                    "release": 0.5,
                    "brightness": 0.8
                }
            ),
            "guitar": Instrument(
                name="guitar",
                type="acoustic",
                parameters={
                    "attack": 0.02,
                    "decay": 0.4,
                    "sustain": 0.6,
                    "release": 0.8,
                    "brightness": 0.7
                }
            ),
            "synth": Instrument(
                name="synth",
                type="synthesizer",
                parameters={
                    "attack": 0.1,
                    "decay": 0.2,
                    "sustain": 0.8,
                    "release": 0.3,
                    "filter_cutoff": 1000,
                    "resonance": 0.5
                }
            ),
            "pad": Instrument(
                name="pad",
                type="synthesizer",
                parameters={
                    "attack": 0.5,
                    "decay": 0.3,
                    "sustain": 0.9,
                    "release": 1.0,
                    "filter_cutoff": 2000,
                    "chorus": 0.3
                }
            )
        }
        return instruments
    
    def get_available_instruments(self) -> List[str]:
        """利用可能な楽器のリストを取得"""
        return list(self.instruments.keys())
    
    def synthesize_sound(self, note: Note, instrument: Instrument, sample_rate: int = 44100) -> AudioClip:
        """音符と楽器に基づいて音を合成"""
        
        # 基本的な音の合成
        frequency = note.to_frequency()
        duration = note.duration
        velocity = note.velocity / 127.0  # 0-1の範囲に正規化
        
        # 時間軸を生成
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # 基本波形を生成
        waveform = self._generate_waveform(frequency, t, instrument)
        
        # ADSR エンベロープを適用
        envelope = self._generate_envelope(t, instrument.get_sound_parameters())
        
        # 音量を適用
        audio_data = waveform * envelope * velocity
        
        return AudioClip(
            audio_data=audio_data,
            sample_rate=sample_rate,
            format="wav"
        )
    
    def _generate_waveform(self, frequency: float, t: np.ndarray, instrument: Instrument) -> np.ndarray:
        """楽器に応じた波形を生成"""
        
        if instrument.type == "acoustic":
            # アコースティック楽器は複数の倍音を持つ
            fundamental = np.sin(2 * np.pi * frequency * t)
            harmonic2 = 0.5 * np.sin(2 * np.pi * frequency * 2 * t)
            harmonic3 = 0.3 * np.sin(2 * np.pi * frequency * 3 * t)
            harmonic4 = 0.2 * np.sin(2 * np.pi * frequency * 4 * t)
            
            return fundamental + harmonic2 + harmonic3 + harmonic4
        
        elif instrument.type == "synthesizer":
            # シンセサイザーは様々な波形を生成可能
            if instrument.name == "synth":
                # ノコギリ波
                return 2 * (t * frequency - np.floor(t * frequency + 0.5))
            elif instrument.name == "pad":
                # 複数のサイン波を重ねた豊かな音
                wave1 = np.sin(2 * np.pi * frequency * t)
                wave2 = 0.7 * np.sin(2 * np.pi * frequency * 1.01 * t)  # わずかにデチューン
                wave3 = 0.5 * np.sin(2 * np.pi * frequency * 2 * t)
                return wave1 + wave2 + wave3
        
        # デフォルトはサイン波
        return np.sin(2 * np.pi * frequency * t)
    
    def _generate_envelope(self, t: np.ndarray, params: Dict) -> np.ndarray:
        """ADSR エンベロープを生成"""
        attack = params.get("attack", 0.1)
        decay = params.get("decay", 0.3)
        sustain = params.get("sustain", 0.7)
        release = params.get("release", 0.5)
        
        total_duration = t[-1]
        envelope = np.ones_like(t)
        
        for i, time in enumerate(t):
            if time < attack:
                # Attack フェーズ
                envelope[i] = time / attack
            elif time < attack + decay:
                # Decay フェーズ
                decay_progress = (time - attack) / decay
                envelope[i] = 1.0 - (1.0 - sustain) * decay_progress
            elif time < total_duration - release:
                # Sustain フェーズ
                envelope[i] = sustain
            else:
                # Release フェーズ
                release_progress = (time - (total_duration - release)) / release
                envelope[i] = sustain * (1.0 - release_progress)
        
        return envelope

# src/services/bgm_generator.py
import os
import numpy as np
from typing import List, Optional
from ..models.user_preference import UserPreference
from ..models.bgm import BGM
from ..models.audio_components import AudioClip, Note, Chord
from .music_theory_engine import MusicTheoryEngine
from .sound_library import SoundLibrary

class BGMGenerator:
    def __init__(self):
        self.music_theory_engine = MusicTheoryEngine()
        self.sound_library = SoundLibrary()
        self.sample_rate = 44100
    
    def generate(self, preference: UserPreference) -> BGM:
        """ユーザー設定に基づいてBGMを生成"""
        
        # 1. メロディを生成
        melody = self.music_theory_engine.generate_melody(preference)
        
        # 2. コード進行を生成
        chords = self.music_theory_engine.generate_chords(melody)
        
        # 3. 楽器を選択
        selected_instruments = self._select_instruments(preference)
        
        # 4. 各パートの音声を生成
        audio_clips = []
        
        # メロディパート
        if "piano" in selected_instruments:
            melody_audio = self._generate_melody_audio(melody, self.sound_library.instruments["piano"])
            audio_clips.append(melody_audio)
        
        # コードパート
        if "guitar" in selected_instruments:
            chord_audio = self._generate_chord_audio(chords, self.sound_library.instruments["guitar"])
            audio_clips.append(chord_audio)
        
        # パッドパート（雰囲気作り）
        if "pad" in selected_instruments:
            pad_audio = self._generate_pad_audio(chords, self.sound_library.instruments["pad"])
            audio_clips.append(pad_audio)
        
        # 5. 音声をミックス
        mixed_audio = self._mix_audio_clips(audio_clips)
        
        # 6. ファイルに保存
        file_path = self._save_audio_file(mixed_audio, preference)
        
        # 7. BGMオブジェクトを作成
        bgm = BGM(
            bgm_id="",  # 自動生成される
            title=self._generate_title(preference),
            file_path=file_path,
            related_preference=preference,
            duration=mixed_audio.duration
        )
        
        return bgm
    
    def _select_instruments(self, preference: UserPreference) -> List[str]:
        """ユーザー設定とジャンルに基づいて楽器を選択"""
        if preference.instruments:
            return preference.instruments
        
        # ジャンルに基づくデフォルト楽器選択
        genre_instruments = {
            "ambient": ["pad", "piano"],
            "electronic": ["synth", "pad"],
            "classical": ["piano"],
            "jazz": ["piano", "guitar"]
        }
        
        return genre_instruments.get(preference.genre, ["piano"])
    
    def _generate_melody_audio(self, melody: List[Note], instrument) -> AudioClip:
        """メロディの音声を生成"""
        audio_segments = []
        
        for note in melody:
            note_audio = self.sound_library.synthesize_sound(note, instrument, self.sample_rate)
            audio_segments.append(note_audio)
        
        return self._concatenate_audio_clips(audio_segments)
    
    def _generate_chord_audio(self, chords: List[Chord], instrument) -> AudioClip:
        """コードの音声を生成"""
        audio_segments = []
        
        for chord in chords:
            # コードの各音を同時に鳴らす
            chord_notes_audio = []
            for note in chord.notes:
                note_audio = self.sound_library.synthesize_sound(note, instrument, self.sample_rate)
                chord_notes_audio.append(note_audio)
            
            # コードの音を重ねる
            chord_audio = self._overlay_audio_clips(chord_notes_audio)
            audio_segments.append(chord_audio)
        
        return self._concatenate_audio_clips(audio_segments)
    
    def _generate_pad_audio(self, chords: List[Chord], instrument) -> AudioClip:
        """パッドの音声を生成（長い音で雰囲気を作る）"""
        # パッドは長い音で和音を演奏
        return self._generate_chord_audio(chords, instrument)
    
    def _mix_audio_clips(self, clips: List[AudioClip]) -> AudioClip:
        """複数の音声クリップをミックス"""
        if not clips:
            return AudioClip(np.array([]), self.sample_rate)
        
        # 最長の音声の長さに合わせる
        max_length = max(len(clip.audio_data) for clip in clips)
        
        # 音声をミックス
        mixed_data = np.
        # src/models/user.py
from dataclasses import dataclass
from typing import Optional
import uuid

@dataclass
class User:
    user_id: str
    user_name: str
    email: Optional[str] = None
    created_at: Optional[str] = None
    
    def __post_init__(self):
        if not self.user_id:
            self.user_id = str(uuid.uuid4())
    
    def request_bgm(self, preferences):
        """BGMをリクエストする"""
        return f"User {self.user_name} requested BGM with preferences: {preferences}"
    
    def send_feedback(self, bgm, rating: int, comment: str = ""):
        """フィードバックを送る"""
        return {
            "user_id": self.user_id,
            "bgm_id": bgm.bgm_id,
            "rating": rating,
            "comment": comment
        }

# src/models/user_preference.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class UserPreference:
    mood: str  # "relaxed", "energetic", "focus", "creative"
    genre: str  # "ambient", "electronic", "classical", "jazz"
    tempo: str  # "slow", "medium", "fast"
    instruments: List[str]  # ["piano", "guitar", "synth", "drums"]
    duration: int  # 再生時間（秒）
    key_signature: Optional[str] = "C"  # 調性
    time_signature: Optional[str] = "4/4"  # 拍子
    
    def validate(self) -> bool:
        """設定の妥当性をチェック"""
        valid_moods = ["relaxed", "energetic", "focus", "creative"]
        valid_genres = ["ambient", "electronic", "classical", "jazz"]
        valid_tempos = ["slow", "medium", "fast"]
        
        return (
            self.mood in valid_moods and
            self.genre in valid_genres and
            self.tempo in valid_tempos and
            self.duration > 0 and
            len(self.instruments) > 0
        )

# src/models/bgm.py
from dataclasses import dataclass
from typing import Optional
import uuid
from datetime import datetime

@dataclass
class BGM:
    bgm_id: str
    title: str
    file_path: str
    related_preference: 'UserPreference'
    created_at: Optional[datetime] = None
    file_size: Optional[int] = None
    duration: Optional[float] = None
    
    def __post_init__(self):
        if not self.bgm_id:
            self.bgm_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now()
    
    def get_metadata(self) -> dict:
        """BGMのメタデータを取得"""
        return {
            "id": self.bgm_id,
            "title": self.title,
            "duration": self.duration,
            "file_size": self.file_size,
            "created_at": self.created_at.isoformat(),
            "preferences": {
                "mood": self.related_preference.mood,
                "genre": self.related_preference.genre,
                "tempo": self.related_preference.tempo,
                "instruments": self.related_preference.instruments
            }
        }

# src/models/audio_components.py
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class Note:
    pitch: str  # "C4", "D#3", etc.
    duration: float  # 音符の長さ（秒）
    velocity: int  # 音の強さ（0-127）
    start_time: float = 0.0
    
    def to_frequency(self) -> float:
        """音高を周波数に変換"""
        # A4 = 440Hz を基準とした計算
        note_map = {
            'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4,
            'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2
        }
        
        note_name = self.pitch[:-1]
        octave = int(self.pitch[-1])
        
        semitones = note_map[note_name] + (octave - 4) * 12
        return 440 * (2 ** (semitones / 12))

@dataclass
class Chord:
    root: str  # ルート音
    quality: str  # "major", "minor", "diminished", "augmented"
    notes: List[Note]
    duration: float = 4.0
    
    @classmethod
    def from_root_and_quality(cls, root: str, quality: str, octave: int = 4):
        """ルート音と品質からコードを生成"""
        intervals = {
            "major": [0, 4, 7],
            "minor": [0, 3, 7],
            "diminished": [0, 3, 6],
            "augmented": [0, 4, 8]
        }
        
        notes = []
        for interval in intervals[quality]:
            # 簡単な実装: 半音階で音程を計算
            notes.append(Note(
                pitch=f"{root}{octave}",
                duration=4.0,
                velocity=80
            ))
        
        return cls(root=root, quality=quality, notes=notes)

@dataclass
class Instrument:
    name: str
    type: str  # "acoustic", "electric", "synthesizer"
    parameters: Dict[str, Any]
    
    def get_sound_parameters(self) -> Dict[str, Any]:
        """楽器の音響パラメータを取得"""
        base_params = {
            "attack": 0.1,
            "decay": 0.3,
            "sustain": 0.7,
            "release": 0.5,
            "volume": 1.0
        }
        base_params.update(self.parameters)
        return base_params

@dataclass
class AudioClip:
    audio_data: np.ndarray
    sample_rate: int
    format: str = "wav"
    
    @property
    def duration(self) -> float:
        """音声の長さを秒で返す"""
        return len(self.audio_data) / self.sample_rate
    
    def normalize(self) -> 'AudioClip':
        """音量を正規化"""
        max_val = np.max(np.abs(self.audio_data))
        if max_val > 0:
            normalized_data = self.audio_data / max_val
        else:
            normalized_data = self.audio_data
        
        return AudioClip(
            audio_data=normalized_data,
            sample_rate=self.sample_rate,
            format=self.format
        )

# src/services/music_theory_engine.py
import random
from typing import List, Dict, Tuple
from ..models.audio_components import Note, Chord
from ..models.user_preference import UserPreference

class MusicTheoryEngine:
    def __init__(self):
        self.scale_patterns = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "pentatonic": [0, 2, 4, 7, 9],
            "blues": [0, 3, 5, 6, 7, 10]
        }
        
        self.chord_progressions = {
            "major": ["I", "V", "vi", "IV"],
            "minor": ["i", "VII", "VI", "VII"],
            "jazz": ["IIM7", "V7", "IM7", "VIM7"]
        }
    
    def generate_melody(self, preference: UserPreference) -> List[Note]:
        """ユーザー設定に基づいてメロディを生成"""
        scale = self._select_scale(preference.genre)
        tempo_bpm = self._determine_tempo(preference.tempo)
        
        melody = []
        current_time = 0.0
        note_duration = 60.0 / tempo_bpm  # 四分音符の長さ
        
        # メロディの長さを決定
        num_notes = int(preference.duration / note_duration)
        
        for i in range(num_notes):
            # スケールからランダムに音を選択
            scale_degree = random.choice(scale)
            octave = random.choice([3, 4, 5])
            
            # 音高を計算
            pitch = self._scale_degree_to_pitch(scale_degree, preference.key_signature, octave)
            
            # 音価をランダムに決定
            duration_multiplier = random.choice([0.5, 1.0, 1.5, 2.0])
            duration = note_duration * duration_multiplier
            
            # 音の強さを決定
            velocity = self._calculate_velocity(preference.mood, i, num_notes)
            
            note = Note(
                pitch=pitch,
                duration=duration,
                velocity=velocity,
                start_time=current_time
            )
            
            melody.append(note)
            current_time += duration
        
        return melody
    
    def generate_chords(self, melody: List[Note]) -> List[Chord]:
        """メロディに基づいてコード進行を生成"""
        chords = []
        
        # 簡単なコード進行パターンを適用
        chord_pattern = ["C", "Am", "F", "G"]  # vi-IV-I-V
        
        measures = len(melody) // 4  # 4つの音符で1小節と仮定
        
        for i in range(measures):
            chord_root = chord_pattern[i % len(chord_pattern)]
            chord = Chord.from_root_and_quality(chord_root, "major")
            chords.append(chord)
        
        return chords
    
    def determine_tempo(self, tempo_preference: str) -> int:
        """テンポ設定からBPMを決定"""
        tempo_map = {
            "slow": random.randint(60, 80),
            "medium": random.randint(90, 120),
            "fast": random.randint(130, 160)
        }
        return tempo_map.get(tempo_preference, 120)
    
    def _select_scale(self, genre: str) -> List[int]:
        """ジャンルに基づいてスケールを選択"""
        genre_scales = {
            "classical": "major",
            "ambient": "minor",
            "jazz": "blues",
            "electronic": "pentatonic"
        }
        
        scale_type = genre_scales.get(genre, "major")
        return self.scale_patterns[scale_type]
    
    def _determine_tempo(self, tempo_preference: str) -> int:
        """テンポ設定からBPMを決定"""
        return self.determine_tempo(tempo_preference)
    
    def _scale_degree_to_pitch(self, scale_degree: int, key: str, octave: int) -> str:
        """スケール度数を音高に変換"""
        chromatic_scale = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        
        # キーのオフセットを計算
        key_offset = chromatic_scale.index(key) if key in chromatic_scale else 0
        
        # 絶対音高を計算
        pitch_class = (key_offset + scale_degree) % 12
        pitch_name = chromatic_scale[pitch_class]
        
        return f"{pitch_name}{octave}"
    
    def _calculate_velocity(self, mood: str, position: int, total_notes: int) -> int:
        """ムードと位置に基づいて音の強さを計算"""
        base_velocity = {
            "relaxed": 60,
            "energetic": 100,
            "focus": 80,
            "creative": 90
        }.get(mood, 80)
        
        # 曲の展開に応じて強弱を調整
        progress = position / total_notes
        if progress < 0.3:  # 序盤
            return base_velocity - 10
        elif progress > 0.7:  # 終盤
            return base_velocity + 10
        else:  # 中盤
            return base_velocity

# src/services/sound_library.py
import numpy as np
from typing import List, Dict
from ..models.audio_components import Instrument, AudioClip, Note

class SoundLibrary:
    def __init__(self):
        self.instruments = self._initialize_instruments()
    
    def _initialize_instruments(self) -> Dict[str, Instrument]:
        """利用可能な楽器を初期化"""
        instruments = {
            "piano": Instrument(
                name="piano",
                type="acoustic",
                parameters={
                    "attack": 0.01,
                    "decay": 0.3,
                    "sustain": 0.7,
                    "release": 0.5,
                    "brightness": 0.8
                }
            ),
            "guitar": Instrument(
                name="guitar",
                type="acoustic",
                parameters={
                    "attack": 0.02,
                    "decay": 0.4,
                    "sustain": 0.6,
                    "release": 0.8,
                    "brightness": 0.7
                }
            ),
            "synth": Instrument(
                name="synth",
                type="synthesizer",
                parameters={
                    "attack": 0.1,
                    "decay": 0.2,
                    "sustain": 0.8,
                    "release": 0.3,
                    "filter_cutoff": 1000,
                    "resonance": 0.5
                }
            ),
            "pad": Instrument(
                name="pad",
                type="synthesizer",
                parameters={
                    "attack": 0.5,
                    "decay": 0.3,
                    "sustain": 0.9,
                    "release": 1.0,
                    "filter_cutoff": 2000,
                    "chorus": 0.3
                }
            )
        }
        return instruments
    
    def get_available_instruments(self) -> List[str]:
        """利用可能な楽器のリストを取得"""
        return list(self.instruments.keys())
    
    def synthesize_sound(self, note: Note, instrument: Instrument, sample_rate: int = 44100) -> AudioClip:
        """音符と楽器に基づいて音を合成"""
        
        # 基本的な音の合成
        frequency = note.to_frequency()
        duration = note.duration
        velocity = note.velocity / 127.0  # 0-1の範囲に正規化
        
        # 時間軸を生成
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # 基本波形を生成
        waveform = self._generate_waveform(frequency, t, instrument)
        
        # ADSR エンベロープを適用
        envelope = self._generate_envelope(t, instrument.get_sound_parameters())
        
        # 音量を適用
        audio_data = waveform * envelope * velocity
        
        return AudioClip(
            audio_data=audio_data,
            sample_rate=sample_rate,
            format="wav"
        )
    
    def _generate_waveform(self, frequency: float, t: np.ndarray, instrument: Instrument) -> np.ndarray:
        """楽器に応じた波形を生成"""
        
        if instrument.type == "acoustic":
            # アコースティック楽器は複数の倍音を持つ
            fundamental = np.sin(2 * np.pi * frequency * t)
            harmonic2 = 0.5 * np.sin(2 * np.pi * frequency * 2 * t)
            harmonic3 = 0.3 * np.sin(2 * np.pi * frequency * 3 * t)
            harmonic4 = 0.2 * np.sin(2 * np.pi * frequency * 4 * t)
            
            return fundamental + harmonic2 + harmonic3 + harmonic4
        
        elif instrument.type == "synthesizer":
            # シンセサイザーは様々な波形を生成可能
            if instrument.name == "synth":
                # ノコギリ波
                return 2 * (t * frequency - np.floor(t * frequency + 0.5))
            elif instrument.name == "pad":
                # 複数のサイン波を重ねた豊かな音
                wave1 = np.sin(2 * np.pi * frequency * t)
                wave2 = 0.7 * np.sin(2 * np.pi * frequency * 1.01 * t)  # わずかにデチューン
                wave3 = 0.5 * np.sin(2 * np.pi * frequency * 2 * t)
                return wave1 + wave2 + wave3
        
        # デフォルトはサイン波
        return np.sin(2 * np.pi * frequency * t)
    
    def _generate_envelope(self, t: np.ndarray, params: Dict) -> np.ndarray:
        """ADSR エンベロープを生成"""
        attack = params.get("attack", 0.1)
        decay = params.get("decay", 0.3)
        sustain = params.get("sustain", 0.7)
        release = params.get("release", 0.5)
        
        total_duration = t[-1]
        envelope = np.ones_like(t)
        
        for i, time in enumerate(t):
            if time < attack:
                # Attack フェーズ
                envelope[i] = time / attack
            elif time < attack + decay:
                # Decay フェーズ
                decay_progress = (time - attack) / decay
                envelope[i] = 1.0 - (1.0 - sustain) * decay_progress
            elif time < total_duration - release:
                # Sustain フェーズ
                envelope[i] = sustain
            else:
                # Release フェーズ
                release_progress = (time - (total_duration - release)) / release
                envelope[i] = sustain * (1.0 - release_progress)
        
        return envelope

# src/services/bgm_generator.py
import os
import numpy as np
from typing import List, Optional
from ..models.user_preference import UserPreference
from ..models.bgm import BGM
from ..models.audio_components import AudioClip, Note, Chord
from .music_theory_engine import MusicTheoryEngine
from .sound_library import SoundLibrary

class BGMGenerator:
    def __init__(self):
        self.music_theory_engine = MusicTheoryEngine()
        self.sound_library = SoundLibrary()
        self.sample_rate = 44100
    
    def generate(self, preference: UserPreference) -> BGM:
        """ユーザー設定に基づいてBGMを生成"""
        
        # 1. メロディを生成
        melody = self.music_theory_engine.generate_melody(preference)
        
        # 2. コード進行を生成
        chords = self.music_theory_engine.generate_chords(melody)
        
        # 3. 楽器を選択
        selected_instruments = self._select_instruments(preference)
        
        # 4. 各パートの音声を生成
        audio_clips = []
        
        # メロディパート
        if "piano" in selected_instruments:
            melody_audio = self._generate_melody_audio(melody, self.sound_library.instruments["piano"])
            audio_clips.append(melody_audio)
        
        # コードパート
        if "guitar" in selected_instruments:
            chord_audio = self._generate_chord_audio(chords, self.sound_library.instruments["guitar"])
            audio_clips.append(chord_audio)
        
        # パッドパート（雰囲気作り）
        if "pad" in selected_instruments:
            pad_audio = self._generate_pad_audio(chords, self.sound_library.instruments["pad"])
            audio_clips.append(pad_audio)
        
        # 5. 音声をミックス
        mixed_audio = self._mix_audio_clips(audio_clips)
        
        # 6. ファイルに保存
        file_path = self._save_audio_file(mixed_audio, preference)
        
        # 7. BGMオブジェクトを作成
        bgm = BGM(
            bgm_id="",  # 自動生成される
            title=self._generate_title(preference),
            file_path=file_path,
            related_preference=preference,
            duration=mixed_audio.duration
        )
        
        return bgm
    
    def _select_instruments(self, preference: UserPreference) -> List[str]:
        """ユーザー設定とジャンルに基づいて楽器を選択"""
        if preference.instruments:
            return preference.instruments
        
        # ジャンルに基づくデフォルト楽器選択
        genre_instruments = {
            "ambient": ["pad", "piano"],
            "electronic": ["synth", "pad"],
            "classical": ["piano"],
            "jazz": ["piano", "guitar"]
        }
        
        return genre_instruments.get(preference.genre, ["piano"])
    
    def _generate_melody_audio(self, melody: List[Note], instrument) -> AudioClip:
        """メロディの音声を生成"""
        audio_segments = []
        
        for note in melody:
            note_audio = self.sound_library.synthesize_sound(note, instrument, self.sample_rate)
            audio_segments.append(note_audio)
        
        return self._concatenate_audio_clips(audio_segments)
    
    def _generate_chord_audio(self, chords: List[Chord], instrument) -> AudioClip:
        """コードの音声を生成"""
        audio_segments = []
        
        for chord in chords:
            # コードの各音を同時に鳴らす
            chord_notes_audio = []
            for note in chord.notes:
                note_audio = self.sound_library.synthesize_sound(note, instrument, self.sample_rate)
                chord_notes_audio.append(note_audio)
            
            # コードの音を重ねる
            chord_audio = self._overlay_audio_clips(chord_notes_audio)
            audio_segments.append(chord_audio)
        
        return self._concatenate_audio_clips(audio_segments)
    
    def _generate_pad_audio(self, chords: List[Chord], instrument) -> AudioClip:
        """パッドの音声を生成（長い音で雰囲気を作る）"""
        # パッドは長い音で和音を演奏
        return self._generate_chord_audio(chords, instrument)
    
    def _mix_audio_clips(self, clips: List[AudioClip]) -> AudioClip:
        """複数の音声クリップをミックス"""
        if not clips:
            return AudioClip(np.array([]), self.sample_rate)
        
        # 最長の音声の長さに合わせる
        max_length = max(len(clip.audio_data) for clip in clips)
        
        # 音声をミックス
        mixed_data = np.zeros(max_length)
        
        for clip in clips:
            # 音声の長さを統一
            padded_data = np.pad(clip.audio_data, (0, max_length - len(clip.audio_data)), mode='constant')
            mixed_data += padded_data
        
        # 音量を正規化
        if np.max(np.abs(mixed_data)) > 0:
            mixed_data = mixed_data / np.max(np.abs(mixed_data)) * 0.8
        
        return AudioClip(mixed_data, self.sample_rate)
    
    def _concatenate_audio_clips(self, clips: List[AudioClip]) -> AudioClip:
        """音声クリップを連結"""
        if not clips:
            return AudioClip(np.array([]), self.sample_rate)
        
        concatenated_data = np.concatenate([clip.audio_data for clip in clips])
        return AudioClip(concatenated_data, self.sample_rate)
    
    def _overlay_audio_clips(self, clips: List[AudioClip]) -> AudioClip:
        """音声クリップを重ねる"""
        if not clips:
            return AudioClip(np.array([]), self.sample_rate)
        
        max_length = max(len(clip.audio_data) for clip in clips)
        overlaid_data = np.zeros(max_length)
        
        for clip in clips:
            padded_data = np.pad(clip.audio_data, (0, max_length - len(clip.audio_data)), mode='constant')
            overlaid_data += padded_data
        
        return AudioClip(overlaid_data, self.sample_rate)
    
    def _save_audio_file(self, audio_clip: AudioClip, preference: UserPreference) -> str:
        """音声ファイルを保存"""
        import wave
        import uuid
        
        # 保存ディレクトリを作成
        os.makedirs("generated_bgm", exist_ok=True)
        
        # ファイル名を生成
        filename = f"bgm_{preference.genre}_{preference.mood}_{uuid.uuid4().hex[:8]}.wav"
        file_path = os.path.join("generated_bgm", filename)
        
        # WAVファイルとして保存
        with wave.open(file_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # モノラル
            wav_file.setsampwidth(2)  # 16ビット
            wav_file.setframerate(self.sample_rate)
            
            # 音声データを16ビット整数に変換
            audio_data_int = np.int16(audio_clip.audio_data * 32767)
            wav_file.writeframes(audio_data_int.tobytes())
        
        return file_path
    
    def _generate_title(self, preference: UserPreference) -> str:
        """設定に基づいてタイトルを生成"""
        mood_adjectives = {
            "relaxed": ["Peaceful", "Serene", "Calm", "Tranquil"],
            "energetic": ["Dynamic", "Vibrant", "Powerful", "Energetic"],
            "focus": ["Focused", "Concentrated", "Clear", "Steady"],
            "creative": ["Inspiring", "Creative", "Imaginative", "Flowing"]
        }
        
        genre_nouns = {
            "ambient": ["Atmosphere", "Soundscape", "Ambience", "Space"],
            "electronic": ["Synthesis", "Digital", "Electronic", "Waves"],
            "classical": ["Melody", "Harmony", "Composition", "Movement"],
            "jazz": ["Improvisation", "Swing", "Blues", "Rhythm"]
        }
        
        adjective = np.random.choice(mood_adjectives.get(preference.mood, ["Beautiful"]))
        noun = np.random.choice(genre_nouns.get(preference.genre, ["Music"]))
        
        return f"{adjective} {noun}"

# src/data/database.py
import sqlite3
import json
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
from ..models.user import User
from ..models.bgm import BGM
from ..models.user_preference import UserPreference

class Database:
    def __init__(self, db_path: str = "bgm_app.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """データベースを初期化"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # ユーザーテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    user_name TEXT NOT NULL,
                    email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # BGMテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bgms (
                    bgm_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    duration REAL,
                    file_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            # ユーザー設定テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    bgm_id TEXT,
                    mood TEXT NOT NULL,
                    genre TEXT NOT NULL,
                    tempo TEXT NOT NULL,
                    instruments TEXT NOT NULL,
                    duration INTEGER NOT NULL,
                    key_signature TEXT DEFAULT 'C',
                    time_signature TEXT DEFAULT '4/4',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (bgm_id) REFERENCES bgms (bgm_id)
                )
            """)
            
            # フィードバックテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    bgm_id TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                    comment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (bgm_id) REFERENCES bgms (bgm_id)
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """データベース接続のコンテキストマネージャー"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def save_user(self, user: User) -> bool:
        """ユーザーを保存"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO users (user_id, user_name, email)
                    VALUES (?, ?, ?)
                """, (user.user_id, user.user_name, user.email))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error saving user: {e}")
            return False
    
    def get_user(self, user_id: str) -> Optional[User]:
        """ユーザーを取得"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                
                if row:
                    return User(
                        user_id=row['user_id'],
                        user_name=row['user_name'],
                        email=row['email'],
                        created_at=row['created_at']
                    )
                return None
        except sqlite3.Error as e:
            print(f"Error getting user: {e}")
            return None
    
    def save_bgm(self, bgm: BGM, user_id: str) -> bool:
        """BGMを保存"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO bgms (bgm_id, title, file_path, user_id, duration, file_size)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (bgm.bgm_id, bgm.title, bgm.file_path, user_id, bgm.duration, bgm.file_size))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error saving BGM: {e}")
            return False
    
    def get_bgms(self, user_id: str) -> List[BGM]:
        """ユーザーのBGMリストを取得"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT b.*, p.mood, p.genre, p.tempo, p.instruments, p.duration as pref_duration,
                           p.key_signature, p.time_signature
                    FROM bgms b
                    LEFT JOIN user_preferences p ON b.bgm_id = p.bgm_id
                    WHERE b.user_id = ?
                    ORDER BY b.created_at DESC
                """, (user_id,))
                
                bgms = []
                for row in cursor.fetchall():
                    # UserPreferenceを再構築
                    preference = UserPreference(
                        mood=row['mood'] or 'relaxed',
                        genre=row['genre'] or 'ambient',
                        tempo=row['tempo'] or 'medium',
                        instruments=json.loads(row['instruments']) if row['instruments'] else ['piano'],
                        duration=row['pref_duration'] or 120,
                        key_signature=row['key_signature'] or 'C',
                        time_signature=row['time_signature'] or '4/4'
                    )
                    
                    bgm = BGM(
                        bgm_id=row['bgm_id'],
                        title=row['title'],
                        file_path=row['file_path'],
                        related_preference=preference,
                        duration=row['duration'],
                        file_size=row['file_size']
                    )
                    bgms.append(bgm)
                
                return bgms
        except sqlite3.Error as e:
            print(f"Error getting BGMs: {e}")
            return []
    
    def save_user_preference(self, user_id: str, preference: UserPreference, bgm_id: str = None) -> bool:
        """ユーザー設定を保存"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO user_preferences 
                    (user_id, bgm_id, mood, genre, tempo, instruments, duration, key_signature, time_signature)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id, bgm_id, preference.mood, preference.genre, preference.tempo,
                    json.dumps(preference.instruments), preference.duration,
                    preference.key_signature, preference.time_signature
                ))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error saving user preference: {e}")
            return False
    
    def get_user_preference(self, user_id: str) -> Optional[UserPreference]:
        """最新のユーザー設定を取得"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM user_preferences 
                    WHERE user_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    return UserPreference(
                        mood=row['mood'],
                        genre=row['genre'],
                        tempo=row['tempo'],
                        instruments=json.loads(row['instruments']),
                        duration=row['duration'],
                        key_signature=row['key_signature'],
                        time_signature=row['time_signature']
                    )
                return None
        except sqlite3.Error as e:
            print(f"Error getting user preference: {e}")
            return None
    
    def save_feedback(self, user_id: str, bgm_id: str, rating: int, comment: str = "") -> bool:
        """フィードバックを保存"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO feedback (user_id, bgm_id, rating, comment)
                    VALUES (?, ?, ?, ?)
                """, (user_id, bgm_id, rating, comment))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error saving feedback: {e}")
            return False
    
    def get_recommendations(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """おすすめBGMを取得"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # 高評価のBGMと似た設定のBGMを推薦
                cursor.execute("""
                    SELECT b.*, p.mood, p.genre, p.tempo, AVG(f.rating) as avg_rating
                    FROM bgms b
                    JOIN user_preferences p ON b.bgm_id = p.bgm_id
                    LEFT JOIN feedback f ON b.bgm_id = f.bgm_id
                    WHERE b.user_id = ?
                    GROUP BY b.bgm_id
                    HAVING avg_rating >= 4.0 OR avg_rating IS NULL
                    ORDER BY avg_rating DESC, b.created_at DESC
                    LIMIT ?
                """, (user_id, limit))
                
                recommendations = []
                for row in cursor.fetchall():
                    recommendations.append({
                        'bgm_id': row['bgm_id'],
                        'title': row['title'],
                        'mood': row['mood'],
                        'genre': row['genre'],
                        'tempo': row['tempo'],
                        'avg_rating': row['avg_rating']
                    })
                
                return recommendations
        except sqlite3.Error as e:
            print(f"Error getting recommendations: {e}")
            return []

# src/main.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from .models.user import User
from .models.user_preference import UserPreference
from .services.bgm_generator import BGMGenerator
from .data.database import Database

app = Flask(__name__)
CORS(app)

# グローバルインスタンス
db = Database()
bgm_generator = BGMGenerator()

@app.route('/api/users', methods=['POST'])
def create_user():
    """ユーザーを作成"""
    try:
        data = request.get_json()
        user = User(
            user_id="",  # 自動生成
            user_name=data['user_name'],
            email=data.get('email')
        )
        
        if db.save_user(user):
            return jsonify({
                'success': True,
                'user_id': user.user_id,
                'message': 'User created successfully'
            }), 201
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to create user'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400

@app.route('/api/users/<user_id>', methods=['GET'])
def get_user(user_id):
    """ユーザー情報を取得"""
    user = db.get_user(user_id)
    if user:
        return jsonify({
            'success': True,
            'user': {
                'user_id': user.user_id,
                'user_name': user.user_name,
                'email': user.email,
                'created_at': user.created_at
            }
        })
    else:
        return jsonify({
            'success': False,
            'message': 'User not found'
        }), 404

@app.route('/api/bgm/generate', methods=['POST'])
def generate_bgm():
    """BGMを生成"""
    try:
        data = request.get_json()
        user_id = data['user_id']
        
        # ユーザー設定を作成
        preference = UserPreference(
            mood=data['mood'],
            genre=data['genre'],
            tempo=data['tempo'],
            instruments=data['instruments'],
            duration=data['duration'],
            key_signature=data.get('key_signature', 'C'),
            time_signature=data.get('time_signature', '4/4')
        )
        
        # 設定を検証
        if not preference.validate():
            return jsonify({
                'success': False,
                'message': 'Invalid preferences'
            }), 400
        
        # BGMを生成
        bgm = bgm_generator.generate(preference)
        
        # データベースに保存
        db.save_bgm(bgm, user_id)
        db.save_user_preference(user_id, preference, bgm.bgm_id)
        
        return jsonify({
            'success': True,
            'bgm': {
                'bgm_id': bgm.bgm_id,
                'title': bgm.title,
                'file_path': bgm.file_path,
                'duration': bgm.duration
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/bgm/<bgm_id>/download', methods=['GET'])
def download_bgm(bgm_id):
    """BGMファイルをダウンロード"""
    try:
        # データベースからBGM情報を取得
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT file_path, title FROM bgms WHERE bgm_id = ?", (bgm_id,))
            row = cursor.fetchone()
            
            if row and os.path.exists(row['file_path']):
                return send_file(
                    row['file_path'],
                    as_attachment=True,
                    download_name=f"{row['title']}.wav"
                )
            else:
                return jsonify({
                    'success': False,
                    'message': 'BGM file not found'
                }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/users/<user_id>/bgms', methods=['GET'])
def get_user_bgms(user_id):
    """ユーザーのBGMリストを取得"""
    bgms = db.get_bgms(user_id)
    return jsonify({
        'success': True,
        'bgms': [bgm.get_metadata() for bgm in bgms]
    })

@app.route('/api/users/<user_id>/recommendations', methods=['GET'])
def get_recommendations(user_id):
    """おすすめBGMを取得"""
    recommendations = db.get_recommendations(user_id)
    return jsonify({
        'success': True,
        'recommendations': recommendations
    })

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """フィードバックを送信"""
    try:
        data = request.get_json()
        success = db.save_feedback(
            user_id=data['user_id'],
            bgm_id=data['bgm_id'],
            rating=data['rating'],
            comment=data.get('comment', '')
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Feedback submitted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to submit feedback'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    """ヘルスチェック"""
    return jsonify({
        'status': 'healthy',
        'message': 'BGM Generator API is running'
    })

if __name__ == '__main__':
    # 生成されたBGMを保存するディレクトリを作成
    os.makedirs("generated_bgm", exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
    // frontend/src/App.js
import React, { useState, useEffect } from 'react';
import './App.css';
import BGMGenerator from './components/BGMGenerator';
import UserPreferences from './components/UserPreferences';
import BGMPlayer from './components/BGMPlayer';
import BGMLibrary from './components/BGMLibrary';
import { createUser, getUser } from './services/api';

function App() {
  const [user, setUser] = useState(null);
  const [currentBGM, setCurrentBGM] = useState(null);
  const [activeTab, setActiveTab] = useState('generator');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // ローカルストレージからユーザー情報を読み込み
    const storedUserId = localStorage.getItem('bgm_user_id');
    if (storedUserId) {
      loadUser(storedUserId);
    }
  }, []);

  const loadUser = async (userId) => {
    try {
      const userData = await getUser(userId);
      setUser(userData);
    } catch (error) {
      console.error('Error loading user:', error);
      localStorage.removeItem('bgm_user_id');
    }
  };

  const handleCreateUser = async (userName, email) => {
    try {
      setLoading(true);
      const userData = await createUser(userName, email);
      setUser(userData);
      localStorage.setItem('bgm_user_id', userData.user_id);
    } catch (error) {
      console.error('Error creating user:', error);
      alert('ユーザー作成に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleBGMGenerated = (bgm) => {
    setCurrentBGM(bgm);
    setActiveTab('player');
  };

  if (!user) {
    return (
      <div className="App">
        <div className="welcome-screen">
          <h1>BGM Generator</h1>
          <p>あなただけのオリジナルBGMを生成します</p>
          <UserRegistration onCreateUser={handleCreateUser} loading={loading} />
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>BGM Generator</h1>
        <p>ようこそ、{user.user_name}さん</p>
      </header>

      <nav className="tab-navigation">
        <button
          className={activeTab === 'generator' ? 'active' : ''}
          onClick={() => setActiveTab('generator')}
        >
          BGM生成
        </button>
        <button
          className={activeTab === 'player' ? 'active' : ''}
          onClick={() => setActiveTab('player')}
        >
          再生
        </button>
        <button
          className={activeTab === 'library' ? 'active' : ''}
          onClick={() => setActiveTab('library')}
        >
          ライブラリ
        </button>
      </nav>

      <main className="App-main">
        {activeTab === 'generator' && (
          <BGMGenerator 
            user={user} 
            onBGMGenerated={handleBGMGenerated}
          />
        )}
        {activeTab === 'player' && (
          <BGMPlayer 
            user={user} 
            bgm={currentBGM} 
            onBGMSelect={setCurrentBGM}
          />
        )}
        {activeTab === 'library' && (
          <BGMLibrary 
            user={user} 
            onBGMSelect={setCurrentBGM}
            onSelectTab={setActiveTab}
          />
        )}
      </main>
    </div>
  );
}

// ユーザー登録コンポーネント
function UserRegistration({ onCreateUser, loading }) {
  const [userName, setUserName] = useState('');
  const [email, setEmail] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (userName.trim()) {
      onCreateUser(userName.trim(), email.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className="user-registration">
      <div className="form-group">
        <label>ユーザー名 *</label>
        <input
          type="text"
          value={userName}
          onChange={(e) => setUserName(e.target.value)}
          required
          placeholder="あなたの名前を入力してください"
        />
      </div>
      <div className="form-group">
        <label>メールアドレス</label>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="example@email.com"
        />
      </div>
      <button type="submit" disabled={loading}>
        {loading ? '作成中...' : '始める'}
      </button>
    </form>
  );
}

export default App;

// frontend/src/components/BGMGenerator.js
import React, { useState } from 'react';
import UserPreferences from './UserPreferences';
import { generateBGM } from '../services/api';

function BGMGenerator({ user, onBGMGenerated }) {
  const [preferences, setPreferences] = useState({
    mood: 'relaxed',
    genre: 'ambient',
    tempo: 'medium',
    instruments: ['piano'],
    duration: 120,
    key_signature: 'C',
    time_signature: '4/4'
  });
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState('');

  const handleGenerate = async () => {
    try {
      setGenerating(true);
      setError('');
      
      const bgm = await generateBGM(user.user_id, preferences);
      onBGMGenerated(bgm);
    } catch (err) {
      setError('BGM生成に失敗しました: ' + err.message);
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="bgm-generator">
      <h2>BGM生成設定</h2>
      
      <UserPreferences 
        preferences={preferences}
        onPreferencesChange={setPreferences}
      />

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      <div className="generation-controls">
        <button 
          onClick={handleGenerate}
          disabled={generating}
          className="generate-button"
        >
          {generating ? 'BGM生成中...' : 'BGMを生成'}
        </button>
      </div>

      {generating && (
        <div className="generation-progress">
          <div className="progress-bar">
            <div className="progress-fill"></div>
          </div>
          <p>あなたの設定に基づいてBGMを作成しています...</p>
        </div>
      )}
    </div>
  );
}

export default BGMGenerator;

// frontend/src/components/UserPreferences.js
import React from 'react';

function UserPreferences({ preferences, onPreferencesChange }) {
  const handleChange = (key, value) => {
    onPreferencesChange({
      ...preferences,
      [key]: value
    });
  };

  const handleInstrumentToggle = (instrument) => {
    const newInstruments = preferences.instruments.includes(instrument)
      ? preferences.instruments.filter(i => i !== instrument)
      : [...preferences.instruments, instrument];
    
    if (newInstruments.length > 0) {
      handleChange('instruments', newInstruments);
    }
  };

  return (
    <div className="user-preferences">
      <div className="preference-section">
        <h3>ムード</h3>
        <div className="radio-group">
          {['relaxed', 'energetic', 'focus', 'creative'].map(mood => (
            <label key={mood}>
              <input
                type="radio"
                name="mood"
                value={mood}
                checked={preferences.mood === mood}
                onChange={(e) => handleChange('mood', e.target.value)}
              />
              {getMoodLabel(mood)}
            </label>
          ))}
        </div>
      </div>

      <div className="preference-section">
        <h3>ジャンル</h3>
        <div className="radio-group">
          {['ambient', 'electronic', 'classical', 'jazz'].map(genre => (
            <label key={genre}>
              <input
                type="radio"
                name="genre"
                value={genre}
                checked={preferences.genre === genre}
                onChange={(e) => handleChange('genre', e.target.value)}
              />
              {getGenreLabel(genre)}
            </label>
          ))}