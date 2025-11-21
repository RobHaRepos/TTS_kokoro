from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    text = '''
    [Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
    '''
    generator = pipeline(
        text, 
        voice='am_onyx'
        )
    
    for i, (gs, ps, audio) in enumerate(generator):
        print(i, gs, ps)
        display(Audio(data=audio, rate=24000, autoplay=i==0))
        sf.write(f'{OUTPUT_DIR}/{i}.wav', audio, 24000)
        
if __name__ == '__main__':
    main()