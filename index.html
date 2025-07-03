import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ScrollView,
  Dimensions,
} from 'react-native';
import * as Tone from 'tone';

const { width, height } = Dimensions.get('window');

const BGMGeneratorApp = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentGenre, setCurrentGenre] = useState('ambient');
  const [currentTempo, setCurrentTempo] = useState(120);
  const [synth, setSynth] = useState(null);
  const [bassLine, setBassLine] = useState(null);
  const [drums, setDrums] = useState(null);
  const [sequence, setSequence] = useState(null);

  // 音色定義
  const genres = {
    ambient: {
      name: 'アンビエント',
      tempo: 80,
      scale: ['C4', 'D4', 'E4', 'G4', 'A4', 'C5', 'D5'],
      chords: [['C4', 'E4', 'G4'], ['F4', 'A4', 'C5'], ['G4', 'B4', 'D5'], ['Am4', 'C5', 'E5']]
    },
    electronic: {
      name: 'エレクトロニック',
      tempo: 128,
      scale: ['C4', 'D4', 'Eb4', 'G4', 'Ab4', 'C5', 'D5'],
      chords: [['C4', 'Eb4', 'G4'], ['F4', 'Ab4', 'C5'], ['Bb4', 'D5', 'F5'], ['Gm4', 'Bb4', 'D5']]
    },
    jazz: {
      name: 'ジャズ',
      tempo: 120,
      scale: ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'Bb4', 'C5'],
      chords: [['C4', 'E4', 'G4', 'B4'], ['F4', 'A4', 'C5', 'E5'], ['G4', 'B4', 'D5', 'F5'], ['Am4', 'C5', 'E5', 'G5']]
    },
    lofi: {
      name: 'Lo-Fi',
      tempo: 85,
      scale: ['C4', 'D4', 'Eb4', 'F4', 'G4', 'Ab4', 'Bb4', 'C5'],
      chords: [['C4', 'Eb4', 'G4'], ['F4', 'Ab4', 'C5'], ['Bb4', 'D5', 'F5'], ['Gm4', 'Bb4', 'D5']]
    }
  };

  // 初期化
  useEffect(() => {
    initializeAudio();
    return () => {
      stopMusic();
    };
  }, []);

  const initializeAudio = async () => {
    try {
      // シンセサイザーの設定
      const synthInstance = new Tone.PolySynth(Tone.Synth, {
        oscillator: {
          type: 'sine'
        },
        envelope: {
          attack: 0.1,
          decay: 0.3,
          sustain: 0.5,
          release: 1
        }
      }).toDestination();

      // ベースラインの設定
      const bassInstance = new Tone.Synth({
        oscillator: {
          type: 'square'
        },
        envelope: {
          attack: 0.1,
          decay: 0.3,
          sustain: 0.4,
          release: 0.8
        }
      }).toDestination();

      // ドラムの設定
      const drumInstance = new Tone.MembraneSynth({
        pitchDecay: 0.05,
        octaves: 10,
        oscillator: {
          type: 'sine'
        },
        envelope: {
          attack: 0.001,
          decay: 0.4,
          sustain: 0.01,
          release: 1.4
        }
      }).toDestination();

      setSynth(synthInstance);
      setBassLine(bassInstance);
      setDrums(drumInstance);

      // 音量調整
      synthInstance.volume.value = -10;
      bassInstance.volume.value = -15;
      drumInstance.volume.value = -20;

    } catch (error) {
      console.error('Audio initialization error:', error);
    }
  };

  const generateMelody = (genre) => {
    const scale = genres[genre].scale;
    const melody = [];
    
    for (let i = 0; i < 8; i++) {
      const note = scale[Math.floor(Math.random() * scale.length)];
      melody.push(note);
    }
    
    return melody;
  };

  const generateBassLine = (genre) => {
    const chords = genres[genre].chords;
    const bassNotes = [];
    
    chords.forEach(chord => {
      // コードのルート音を1オクターブ下げる
      const rootNote = chord[0].replace(/\d/, (match) => (parseInt(match) - 1).toString());
      bassNotes.push(rootNote);
    });
    
    return bassNotes;
  };

  const startMusic = async () => {
    try {
      if (Tone.context.state !== 'running') {
        await Tone.start();
      }

      const genre = genres[currentGenre];
      Tone.Transport.bpm.value = genre.tempo;

      const melody = generateMelody(currentGenre);
      const bass = generateBassLine(currentGenre);

      // メロディーシーケンス
      const melodySequence = new Tone.Sequence((time, note) => {
        if (synth) {
          synth.triggerAttackRelease(note, '8n', time);
        }
      }, melody, '8n');

      // ベースラインシーケンス
      const bassSequence = new Tone.Sequence((time, note) => {
        if (bassLine) {
          bassLine.triggerAttackRelease(note, '4n', time);
        }
      }, bass, '2n');

      // ドラムパターン
      const drumPattern = new Tone.Sequence((time, note) => {
        if (drums) {
          drums.triggerAttackRelease('C2', '16n', time);
        }
      }, ['C2', null, 'C2', null], '8n');

      melodySequence.start(0);
      bassSequence.start(0);
      drumPattern.start(0);

      setSequence([melodySequence, bassSequence, drumPattern]);
      
      Tone.Transport.start();
      setIsPlaying(true);

    } catch (error) {
      console.error('Error starting music:', error);
      Alert.alert('エラー', '音楽の再生に失敗しました');
    }
  };

  const stopMusic = () => {
    try {
      Tone.Transport.stop();
      Tone.Transport.cancel();
      
      if (sequence) {
        sequence.forEach(seq => {
          if (seq) {
            seq.dispose();
          }
        });
        setSequence(null);
      }
      
      setIsPlaying(false);
    } catch (error) {
      console.error('Error stopping music:', error);
    }
  };

  const togglePlayback = () => {
    if (isPlaying) {
      stopMusic();
    } else {
      startMusic();
    }
  };

  const changeGenre = (genre) => {
    if (isPlaying) {
      stopMusic();
    }
    setCurrentGenre(genre);
    setCurrentTempo(genres[genre].tempo);
  };

  const generateNewPattern = () => {
    if (isPlaying) {
      stopMusic();
      setTimeout(() => {
        startMusic();
      }, 100);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>🎵 BGM Generator</Text>
        <Text style={styles.subtitle}>自動音楽生成アプリ</Text>
      </View>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {/* 再生コントロール */}
        <View style={styles.controlSection}>
          <TouchableOpacity
            style={[styles.playButton, isPlaying && styles.playButtonActive]}
            onPress={togglePlayback}
          >
            <Text style={styles.playButtonText}>
              {isPlaying ? '⏸️ 停止' : '▶️ 再生'}
            </Text>
          </TouchableOpacity>
          
          <TouchableOpacity
            style={styles.generateButton}
            onPress={generateNewPattern}
          >
            <Text style={styles.generateButtonText}>🎲 新しいパターン</Text>
          </TouchableOpacity>
        </View>

        {/* ジャンル選択 */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>ジャンル選択</Text>
          <View style={styles.genreContainer}>
            {Object.entries(genres).map(([key, genre]) => (
              <TouchableOpacity
                key={key}
                style={[
                  styles.genreButton,
                  currentGenre === key && styles.genreButtonActive
                ]}
                onPress={() => changeGenre(key)}
              >
                <Text style={[
                  styles.genreButtonText,
                  currentGenre === key && styles.genreButtonTextActive
                ]}>
                  {genre.name}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* 現在の設定 */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>現在の設定</Text>
          <View style={styles.infoContainer}>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>ジャンル:</Text>
              <Text style={styles.infoValue}>{genres[currentGenre].name}</Text>
            </View>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>テンポ:</Text>
              <Text style={styles.infoValue}>{currentTempo} BPM</Text>
            </View>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>状態:</Text>
              <Text style={[styles.infoValue, { color: isPlaying ? '#4CAF50' : '#FF5722' }]}>
                {isPlaying ? '再生中' : '停止中'}
              </Text>
            </View>
          </View>
        </View>

        {/* 使用方法 */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>使用方法</Text>
          <Text style={styles.instructionText}>
            1. お好みのジャンルを選択してください{'\n'}
            2. 「再生」ボタンで音楽を開始{'\n'}
            3. 「新しいパターン」で別のメロディーを生成{'\n'}
            4. 各ジャンルごとに異なる雰囲気の音楽が楽しめます
          </Text>
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a1a',
  },
  header: {
    paddingTop: 60,
    paddingBottom: 20,
    paddingHorizontal: 20,
    backgroundColor: '#2c2c2c',
    borderBottomWidth: 2,
    borderBottomColor: '#444',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 14,
    color: '#ccc',
    textAlign: 'center',
    marginTop: 5,
  },
  content: {
    flex: 1,
    paddingHorizontal: 20,
  },
  controlSection: {
    marginTop: 30,
    marginBottom: 20,
  },
  playButton: {
    backgroundColor: '#4CAF50',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 25,
    alignItems: 'center',
    marginBottom: 15,
  },
  playButtonActive: {
    backgroundColor: '#FF5722',
  },
  playButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  generateButton: {
    backgroundColor: '#2196F3',
    paddingVertical: 12,
    paddingHorizontal: 25,
    borderRadius: 20,
    alignItems: 'center',
  },
  generateButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  section: {
    marginBottom: 25,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 15,
  },
  genreContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  genreButton: {
    backgroundColor: '#3c3c3c',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 15,
    marginBottom: 10,
    width: '48%',
    alignItems: 'center',
  },
  genreButtonActive: {
    backgroundColor: '#FF9800',
  },
  genreButtonText: {
    color: '#ccc',
    fontSize: 14,
    fontWeight: '600',
  },
  genreButtonTextActive: {
    color: '#fff',
  },
  infoContainer: {
    backgroundColor: '#2c2c2c',
    padding: 15,
    borderRadius: 10,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  infoLabel: {
    color: '#ccc',
    fontSize: 14,
  },
  infoValue: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  instructionText: {
    color: '#ccc',
    fontSize: 14,
    lineHeight: 20,
    backgroundColor: '#2c2c2c',
    padding: 15,
    borderRadius: 10,
  },
});

export default BGMGeneratorApp;