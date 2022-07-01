import os
import streamlit as st
import streamlit.components.v1 as components
import requests
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot  as plt 
import tensorflow as tf
import tensorflow_hub as hub
from keras.models import load_model
from PIL import Image
from pydub import AudioSegment
from matplotlib.colors import Normalize
from PIL import Image, ImageOps
from streamlit_lottie import st_lottie


st.set_page_config(page_title="Audio Classification")

st.markdown('''<style>.css-1egvi7u {margin-top: -3rem;}</style>''',
    unsafe_allow_html=True)

st.markdown('''<style>.stAudio {height: 45px;}</style>''',
    unsafe_allow_html=True)

st.markdown('''<style>.css-v37k9u a {color: #ff4c4b;}</style>''',
    unsafe_allow_html=True)  # darkmode
st.markdown('''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''',
    unsafe_allow_html=True)  # lightmode


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
    
lottie_url_sidebar = "https://assets1.lottiefiles.com/packages/lf20_g8zezdhx.json"
lottie_sidebar = load_lottieurl(lottie_url_sidebar)
pngfile = 'generated_spec.png'
    
def get_label_for_audio(audio_num):
    labelarray = ['Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar'
                 'Aircraft', 'Alarm', 'Animal', 'Applause', 'Bark', 'Bass_drum', 'Bass_guitar'
                 'Bathtub_filling_or_washing', 'Bell', 'Bicycle', 'Bicycle_bell', 'Bird'
                 'Bird_vocalization_and_bird_call_and_bird_song', 'Boat_and_Water_vehicle'
                 'Boiling', 'Boom', 'Bowed_string_instrument', 'Brass_instrument', 'Breathing'
                 'Burping_and_eructation', 'Bus', 'Buzz', 'Camera', 'Car', 'Car_passing_by'
                 'Cat', 'Chatter', 'Cheering', 'Chewing_and_mastication'
                 'Chicken_and_rooster', 'Child_speech_and_kid_speaking', 'Chime'
                 'Chink_and_clink', 'Chirp_and_tweet', 'Chuckle_and_chortle', 'Church_bell'
                 'Clapping', 'Clock', 'Coin_dropping', 'Computer_keyboard', 'Conversation'
                 'Cough', 'Cowbell', 'Crack', 'Crackle', 'Crash_cymbal', 'Cricket', 'Crow'
                 'Crowd', 'Crumpling_and_crinkling', 'Crushing', 'Crying_and_sobbing'
                 'Cupboard_open_or_close', 'Cutlery_and_silverware', 'Cymbal'
                 'Dishes_and_pots_and_pans', 'Dog', 'Domestic_sounds_and_home_sounds', 'Door'
                 'Doorbell', 'Drawer_open_or_close', 'Drill', 'Drip', 'Drum', 'Drum_kit'
                 'Electric_guitar', 'Engine', 'Engine_starting', 'Explosion', 'Fart'
                 'Female_singing', 'Female_speech_and_woman_speaking', 'Fill_with_liquid'
                 'Finger_snapping', 'Fire', 'Fireworks', 'Fixed-wing_aircraft_and_airplane'
                 'Fowl', 'Frog', 'Frying_food', 'Gasp', 'Giggle', 'Glass', 'Glockenspiel', 'Gong'
                 'Growling', 'Guitar', 'Gull_and_seagull', 'Gunshot_and_gunfire', 'Gurgling'
                 'Hammer', 'Hands', 'Harmonica', 'Harp', 'Hi-hat', 'Hiss', 'Human_group_actions'
                 'Human_voice', 'Idling', 'Insect', 'Keyboard_musical', 'Keys_jangling'
                 'Knock', 'Laughter', 'Liquid'
                 'Livestock_and_farm_animals_and_working_animals', 'Male_singing'
                 'Male_speech_and_man_speaking', 'Mallet_percussion'
                 'Marimba_and_xylophone', 'Mechanical_fan', 'Mechanisms', 'Meow'
                 'Microwave_oven', 'Motor_vehicle_road', 'Motorcycle', 'Musical_instrument'
                 'Ocean', 'Organ', 'Packing_tape_and_duct_tape', 'Percussion', 'Piano'
                 'Plucked_string_instrument', 'Power_tool', 'Printer', 'Purr'
                 'Race_car_and_auto_racing', 'Rain', 'Raindrop', 'Ratchet_and_pawl', 'Rattle'
                 'Rattle_instrument', 'Respiratory_sounds', 'Ringtone', 'Run', 'Sawing'
                 'Scissors', 'Scratching_performance_technique', 'Screaming', 'Screech'
                 'Shatter', 'Shout', 'Sigh', 'Singing', 'Sink_filling_or_washing', 'Siren'
                 'Skateboard', 'Slam', 'Sliding_door', 'Snare_drum', 'Sneeze', 'Speech'
                 'Speech_synthesizer', 'Splash_and_splatter', 'Squeak', 'Stream', 'Strum'
                 'Subway_and_metro_and_underground', 'Tabla', 'Tambourine', 'Tap', 'Tearing'
                 'Telephone', 'Thump_and_thud', 'Thunder', 'Thunderstorm', 'Tick', 'Tick-tock'
                 'Toilet_flush', 'Tools', 'Traffic_noise_and_roadway_noise', 'Train'
                 'Trickle_and_dribble', 'Truck', 'Trumpet', 'Typewriter', 'Typing', 'Vehicle'
                 'Vehicle_horn_and_car_horn_and_honking', 'Walk_and_footsteps', 'Water'
                 'Water_tap_and_faucet', 'Waves_and_surf', 'Whispering'
                 'Whoosh_and_swoosh_and_swish', 'Wild_animals', 'Wind', 'Wind_chime'
                 'Wind_instrument_and_woodwind_instrument', 'Wood', 'Writing', 'Yell'
                 'Zipper_clothing', 'air_conditioner', 'car_horn', 'children_playing'
                 'clapping', 'coughing', 'dog_bark', 'door_bells', 'door_open_close'
                 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'kitchen_exhaust'
                 'microwave beeping', 'microwave_beeping', 'person_walking', 'siren'
                 'street_music', 'television _on', 'television_on', 'water_faucet']
                 
    if not audio_num >= len(labelarray):            
        return labelarray[audio_num]
    else:
        return None
  
def spec_to_image(spec, eps=1e-6):
  mean = spec.mean()
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps)
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
  spec_scaled = spec_scaled.astype(np.uint8)
  return spec_scaled
  
def generate_melspec(file):
  filect = 0
  rowcount= 1
  FILE_SIZE = 1   #Length of each sample in seconds
  SAMPLING_RATE = 44100
  filect = filect+1
  currpct = 1
  #signal, sr = librosa.load(file, sr=None)
  signal, sr = librosa.load(file, sr=SAMPLING_RATE, mono=True)
  if signal.shape[0]<FILE_SIZE*sr:
    signal=np.pad(signal,int(np.ceil((FILE_SIZE*sr-signal.shape[0])/2)),mode='reflect')
  else:
    signal=signal[:FILE_SIZE*sr]
  librosa.effects.split(signal, top_db=10, frame_length=1000, hop_length=512)
  # my parameters
  hop_length = 512
  n_fft = 2048
  fmin = 20
  fmax = 8300
  top_db = 80
  n_mels = 128
  hop_length_duration = float(hop_length) / sr
  n_fft_duration = float(n_fft) / sr
  # Mel filter banks
  filter_banks = librosa.filters.mel(n_fft=n_fft, sr=sr, n_mels=n_mels)
  stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
  # This will yield 96-length spectrograms
  spectrogram = np.abs(stft)
  # converted to 64 bin log-scaled Mel spectrogram,
  melspec = librosa.feature.melspectrogram(y=signal, sr=sr, S=spectrogram, n_fft=n_fft, hop_length=hop_length,
                                             win_length=960, window='hann',  n_mels=n_mels, fmin=fmin, fmax=fmax, htk=True)
  log_mel_spectrogram = librosa.power_to_db(melspec, top_db=top_db)
  scaled_log_mel_spectogram = spec_to_image(log_mel_spectrogram)
  fig = librosa.display.specshow(scaled_log_mel_spectogram, hop_length=hop_length)
  fig2 = plt.gcf()
  plt.axis('off')
  fig2.set_size_inches(2.9, 2.98)
  plt.savefig(pngfile, dpi=100, bbox_inches='tight', pad_inches=0, format='png')
  img = Image.open(pngfile)
  img = img.resize((224, 224), Image.ANTIALIAS)
  img.save(pngfile, optimize=True)

def main_func():
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
    st_audiorec = components.declare_component("st_audiorec", path=build_dir)

    # TITLE and Creator information
    st.markdown(f'<h1 style="color:#05c0a7;font-size:45px;">Machine Learning Model for Audio Classification</h1>', unsafe_allow_html=True)
    st.markdown('Start recording. Download. Select the downloaded file in the file upload.')
    st.write('\n\n')
    
    # SIDEBAR information
    with st.sidebar:
        st.markdown(f'<h1 style="color:#04bfda;font-size:25px;">MobileNet V2</h1>', unsafe_allow_html=True)
        st_lottie(lottie_sidebar, key="sidebar")
        st.markdown("<label style='text-align: justify; color: black;'>&emsp; &emsp; The model is trained using MobileNet V2, which is a family of neural network architectures for efficient on-device image classification and related tasks. The model uses spectogram image generated from the audio file to classify the sound. The file must be in .wav format.</label>", unsafe_allow_html=True)


    st_audiorec()
    audio_bytes = st.file_uploader("Upload your audio", type=['wav'])
    if audio_bytes is not None:
        model = load_model('mobilenetv2_headless_ftune_final4_LLR.h5', custom_objects={'KerasLayer': hub.KerasLayer})
        generate_melspec(audio_bytes)
        st.markdown(f'<h1 style="color:#05c0a7;font-size:30px;">Uploaded File:</h1>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')

        with col2:
            st.image(pngfile, caption='Spectrogram of the uploaded .wav file', width = 300)

        with col3:
            st.write(' ')
        image = Image.open(pngfile).convert('RGB')
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        size = (224, 224)
        image_batch = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        result = get_label_for_audio(np.argmax(prediction))
        if not result == None:
            col2.markdown("<label style='text-align: center; color: black; font-size: 20px; font-weight: italic; '>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; The audio is from " + result + " </label>", unsafe_allow_html=True)
        
if __name__ == '__main__':

    main_func()
