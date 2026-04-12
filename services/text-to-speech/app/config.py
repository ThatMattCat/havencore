import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory for generated audio (written then read back into the HTTP response)
AUDIO_DIR = os.path.join(BASE_DIR, "output")

################ Solo mode (single container outside of docker compose project) #######
SOLO = False

if SOLO:
    LANGUAGE = "a"
    VOICE = "af_heart"
    MODEL_DEVICE = "cpu"  # To use graphics cards, use something like: "cuda:0"

############################ Do not make changes below this line ######################
else:
    import shared.configs.shared_config as shared_config
    LANGUAGE = shared_config.TTS_LANGUAGE if shared_config.TTS_LANGUAGE else "a"
    VOICE = shared_config.TTS_VOICE if shared_config.TTS_VOICE else "af_heart"
    MODEL_DEVICE = shared_config.TTS_DEVICE if shared_config.TTS_DEVICE else "cpu"  # eg: "cuda:0"
