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

# Word -> inline misaki phoneme override (no surrounding braces — added by the
# preprocessor). Kokoro 0.9.4's misaki G2P treats {phonemes} segments as raw
# phonemes instead of running them through grapheme-to-phoneme.
# Default forces "Selene" to render as 2-syllable "Suh-LEEN" (homophone with
# "Celine"); misaki otherwise reads it as 3-syllable "Sell-uh-nee".
if not SOLO:
    _agent_name = shared_config.AGENT_NAME
else:
    _agent_name = "Selene"

TTS_PRONUNCIATIONS = {
    _agent_name: os.getenv("TTS_AGENT_NAME_PHONEMES", "səˈlin"),
}
