import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 



# Common
AUDIO_DIR = os.path.join(BASE_DIR, "output")
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 6003

################ Solo mode (single container outside of docker compose project) #######
SOLO = False

#### These can be changed if running SOLO
if SOLO:
    LANGUAGE = "a"
    VOICE = "af_heart"
    BASE_URL = "http://localhost:6003/"
    MODEL_DEVICE = "cpu" # To use graphics cards, use something like: "cuda:0"

############################ Do not make changes below this line ######################


######### Important for default behavior in HavenCore Project, ignore if in SOLO mode
else:
    #from shared.scripts.trace_id import with_trace, get_trace_id
    import shared.configs.shared_config as shared_config
    LANGUAGE = shared_config.TTS_LANGUAGE if shared_config.TTS_LANGUAGE else "a"
    VOICE = shared_config.TTS_VOICE if shared_config.TTS_VOICE else "af_heart"
    BASE_URL = f"http://{shared_config.HOST_IP_ADDRESS}:{SERVER_PORT}/" if shared_config.HOST_IP_ADDRESS else "http://localhost:6003/"
    MODEL_DEVICE = shared_config.TTS_DEVICE if shared_config.TTS_DEVICE else "cpu" #eg: "cuda:0"