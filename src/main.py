import torch
import torchaudio
import os
from pprint import pprint
from omegaconf import OmegaConf
from pydub import AudioSegment
from pydub.playback import play
from datetime import datetime
from pathlib import Path
import sys
import re
import numpy as np
import mistral_client

def generateVoice(text, outFile):
    language = 'ru'
    model_id = 'v5_ru'
    device = torch.device('cpu')

    model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                        model='silero_tts',
                                        language=language,
                                        speaker=model_id,
                                        progress = True)
    model.to(device)  # gpu or cpu
    model.speakers

    sample_rate = 48000
    #голос: 'aidar', 'baya', 'kseniya', 'xenia', 'random'
    speaker = 'xenia'
    put_accent=True
    put_yo=True
    put_stress_homo=True
    put_yo_homo=True

    example_text = text

    audio = model.apply_tts(ssml_text=example_text,
                            speaker=speaker,
                            sample_rate=sample_rate,
                            put_accent=put_accent,
                            put_yo=put_yo,
                            put_stress_homo=put_stress_homo,
                            put_yo_homo=put_yo_homo)
    audio_paths = model.save_wav(ssml_text=example_text,
                                speaker=speaker,
                                sample_rate=sample_rate)
    
    #torchaudio.save(outFile, audio.unsqueeze(0), sample_rate)


def mix_audio(voiceFile, backgroundMusic, outputFileName):
    backgound = AudioSegment.from_file(backgroundMusic, format="mp3")
    voice = AudioSegment.from_file(voiceFile)

    # Выравнивание длительности (если нужно)
    backgound = backgound[:120000]  

    # Наложение (смешивание) с регулировкой громкости
    mixed = backgound.overlay(voice, position=0, gain_during_overlay=-10)

    # Сохранение результата
    mixed.export(outputFileName, format="mp3")
    # play(mixed)

def split_text_by_sentences(text, max_size):
    """Разделяет текст, стараясь сохранить целостность предложений"""
    # Разбиваем текст на предложения
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        # +2 для точки и пробела
        if current_size + len(sentence) + 2 <= max_size:
            current_chunk.append(sentence)
            current_size += len(sentence) + 2
        else:
            if current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_size = len(sentence)
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks

def split_ssml_text(text, max_chunk_size=1000):
    """
    Split SSML text into smaller chunks while preserving SSML structure.
    Splits by paragraphs (<p> tags) to maintain natural breaks.
    """
    #убираю пробелы
    cleaned = re.sub(r'\s+', ' ', text)
    text =  cleaned.strip()
    #убираю перенос строк
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Extract content within <speak> tags
    speak_match = re.search(r'<speak>(.*?)</speak>', text, re.DOTALL)
    if not speak_match:
        # If no <speak> tags, treat as plain text
        print("no speak tag. no parsing")
        return [text]
    
    content = speak_match.group(1)
    
    # Split by paragraph tags
    paragraphs = re.findall(r'<p>.*?</p>|<break\s+time="[^"]+"\s*/>', content, re.DOTALL)
    
    chunks = []
    current_chunk = []
    current_length = 0
    max_chunk_size = max_chunk_size - 22
    
    for para in paragraphs:
        para_length = len(para)
        
        # If single paragraph is too long, we still need to include it
        # but as a separate chunk
        if para_length > max_chunk_size:
            if current_chunk:
                chunks.append('c' + ''.join(current_chunk) + '</speak>')
                current_chunk = []
                current_length = 0
            #TODO: разделение слишком длинного параграфа
            chunks.append('<speak>' + para + '</speak>')
            subP = split_text_by_sentences(para , max_chunk_size)
        elif current_length + para_length > max_chunk_size:
            # Save current chunk and start new one
            chunks.append('<speak>' + ''.join(current_chunk) + '</speak>')
            current_chunk = [para]
            current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length
    
    # Add remaining chunk
    if current_chunk:
        chunks.append('<speak>' + ''.join(current_chunk) + '</speak>')
    
    return chunks if chunks else [text]


def generateSsmlText(text, output_dir='out', speaker = "baya"):
    """
    Generate audio from SSML text, handling long texts by chunking.
    Returns the path to the final audio file.
    """
    device = torch.device('cpu')
    torch.set_num_threads(4)
    local_file = 'I:/git/meditation_bot/resources/models/v5_ru.pt'

    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v5_ru.pt',
                                    local_file)  

    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)

    sample_rate = 48000
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split text into chunks
    chunks = split_ssml_text(text, max_chunk_size=1000)
    print(f"Split text into {len(chunks)} chunks")
    
    if len(chunks) == 1:
        # Single chunk - process normally
        audio_paths = model.save_wav(ssml_text=text,
                                    speaker=speaker,
                                    sample_rate=sample_rate)
        return audio_paths
    else:
        # Multiple chunks - process each and combine
        audio_segments = []
        temp_files = []
        
        for i, chunk in enumerate(chunks):

            print(f"Processing chunk {i+1}/{len(chunks)}. Len of chunk is {len(chunk)}...")
            try:
                audio_path = model.save_wav(ssml_text=chunk,
                                          speaker=speaker,
                                          sample_rate=sample_rate)
                temp_files.append(audio_path)
                audio_segments.append(AudioSegment.from_file(audio_path))
            except Exception as e:
                print(f"Error processing chunk {i+1}: {e}")
                continue
        
        # Combine all audio segments
        if audio_segments:
            print("Combining audio chunks...")
            combined_audio = audio_segments[0]
            for segment in audio_segments[1:]:
                combined_audio += segment
            
            # Save combined audio
            now = datetime.now()
            combined_path = os.path.join(output_dir, f"voice_{now.strftime('%Y-%m-%d_%H-%M-%S')}.wav")
            combined_audio.export(combined_path, format="wav")
            
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except:
                    pass
            
            print(f"Combined audio saved to: {combined_path}")
            return combined_path
        else:
            raise RuntimeError("Failed to process any chunks")

#вывод доступных моделей в консоль
def showModels():
    torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                               'latest_silero_models.yml',
                               progress=False)
    models = OmegaConf.load('latest_silero_models.yml')
    available_languages = list(models.tts_models.keys())
    print(f'Available languages {available_languages}')

    for lang in available_languages:
        _models = list(models.tts_models.get(lang).keys())
        print(f'Available models for {lang}: {_models}')

def readFile(file_path): 
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        return text

def generateFileNames():
    now = datetime.now()
    voiceFileName = f"out/voice_{now.strftime('%Y-%m-%d_%H-%M-%S')}.mp3"
    outFileName = f"out/mix_{now.strftime('%Y-%m-%d_%H-%M-%S')}.mp3"
    return (voiceFileName, outFileName)

def getTextFromMistral() -> str:
    client = mistral_client.MistralClient()
    system_prompt = "сгенерируй текст для медитации на основе ответа пользователя, как он себя чувствует. Ответ в формате ssml, без дополнительных комментариев"
    prompt = "Сегодня был тяжелый день, хочу отдохнуть."
    response = client.ask(prompt=prompt, system_promt=system_prompt, max_tokens=1000)
    print("Mistral response:")
    print(response)
    return response

def getSsmlTest(fileName : str):
    if fileName is None:
        return getTextFromMistral()
    file_path = "I:/src/meditation_bot/resources/text/ssml_900.txt"
    print(f"Processing text file {file_path}")
    ssmlText = readFile(file_path)
    print(f"Text length: {len(ssmlText)} characters")

if __name__ == "__main__":
    ssmlText = getSsmlTest(None)
    #костыль для решения проблемы с ffmpeg
    sys.path.append('C:/Program Files/ffmpeg/bin')
    
    # Generate voice audio (handles chunking automatically)
    voice = generateSsmlText(ssmlText)

    print(f"Voice file generated: {voice}")
    
    # Generate output filename for mixed audio
    now = datetime.now()
    outFileName = f"out/mix_{now.strftime('%Y-%m-%d_%H-%M-%S')}.mp3"
    
    # Overlay with background music
    backgroundMusic = "I:/git/meditation_bot/resources/music/001.mp3"
    mix_audio(voice, backgroundMusic, outFileName)
    
    print(f"Final result saved to: {outFileName}")

